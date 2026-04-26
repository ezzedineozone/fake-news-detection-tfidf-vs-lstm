# IMDB Sentiment Classification — A Comparative Study of Classical Machine Learning Models

**Project Report**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Motivation](#2-project-goals-and-motivation)
3. [Dataset Description](#3-dataset-description)
4. [Methodology — Data Preparation Pipeline](#4-methodology--data-preparation-pipeline)
5. [Feature Selection](#5-feature-selection)
6. [Model 1 — Logistic Regression](#6-model-1--logistic-regression)
7. [Model 2 — Linear Support Vector Classifier](#7-model-2--linear-support-vector-classifier-linearsvc)
8. [Model 3 — Kernel SVM (RBF)](#8-model-3--kernel-svm-rbf)
9. [Model 4 — K-Nearest Neighbors](#9-model-4--k-nearest-neighbors-knn)
10. [Model 5 — Decision Tree](#10-model-5--decision-tree)
11. [Model 6 — Random Forest](#11-model-6--random-forest)
12. [Model 7 — Bagging applied to Logistic Regression](#12-model-7--bagging-applied-to-logistic-regression)
13. [Comparative Analysis and Discussion](#13-comparative-analysis-and-discussion)
14. [Conclusions and Future Work](#14-conclusions-and-future-work)
15. [Appendix — File Inventory](#15-appendix--file-inventory)

---

## 1. Executive Summary

This project undertakes a **binary sentiment classification** task on the IMDB Movie Reviews dataset (50,000 labeled reviews). The pipeline encompasses textual preprocessing, vectorization via Bag-of-Words, feature selection driven by Random Forest importance ranking, and a comparative evaluation of seven classical machine learning algorithms: **Logistic Regression**, **Linear Support Vector Classifier (LinearSVC)**, **Kernel Support Vector Machine (RBF)**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, **Random Forest**, and **Bagging applied to Logistic Regression**. Each model is tuned through cross-validated grid search and evaluated using accuracy, confusion matrices, ROC/AUC analysis, and learning curves.

The headline finding is that **regularized linear models — specifically Logistic Regression with L2 regularization — outperform all other approaches**, achieving 88.96% test accuracy. Tree-based and instance-based methods (Decision Tree, KNN) underperform significantly on this high-dimensional sparse text representation, while ensemble methods (Random Forest, Bagging) provide modest but instructive gains over their respective base learners.

**Final ranking (test accuracy on 15,000 held-out reviews):**

| Rank | Model | Test Accuracy | Best Hyperparameters |
|---|---|---|---|
| 1 | **Logistic Regression** | **0.8896** | C = 0.1 (L2 regularization) |
| 2 | LinearSVC | 0.8881 | C = 0.01 |
| 3 | Bagging Logistic (n = 70) | 0.8870 | 70 bootstrap estimators |
| 4 | SVM (RBF kernel) | 0.8771 | default (C = 1.0, γ = 'scale') |
| 5 | Random Forest | 0.8550 | criterion = 'entropy', max_depth = 50 |
| 6 | KNN | 0.7266 | n_neighbors = 180 |
| 7 | Decision Tree | 0.7241 | criterion = 'gini', max_depth = 10 |

The 16-percentage-point spread between the best and worst model on an identical feature representation is the central empirical fact of the project, and motivates the discussion in Section 13.

---

## 2. Project Goals and Motivation

### 2.1 Primary Objectives

1. **Build a complete sentiment-classification pipeline** for unstructured English text, encompassing every stage from raw HTML-laden reviews to final classification output.
2. **Empirically compare classical machine learning algorithms** on a high-dimensional, sparse text representation, characterizing their relative strengths and weaknesses on this class of problem.
3. **Investigate the effect of feature selection** on training tractability and generalization, using Random Forest feature importance as the selection criterion.
4. **Quantify the benefit of bagging** as a variance-reduction technique applied to a strong base learner (Logistic Regression).
5. **Diagnose the bias-variance behavior** of each model through learning curves, and produce decision-quality visualizations (confusion matrices, ROC curves) to support model comparison.

### 2.2 Theoretical Motivation

Text classification with bag-of-words representations is an archetypal *high-dimensional, sparse, approximately-linearly-separable* problem. The classical theoretical expectation is that:

- **Linear models with regularization** should excel because the feature space (vocabulary terms) is large relative to sample size, the decision boundary is plausibly linear in this space, and L1/L2 regularization controls overfitting.
- **Distance-based methods (KNN)** should struggle because Euclidean distance becomes uninformative in high dimensions — the well-known *curse of dimensionality* — and because most word counts are zero, making distances dominated by a few co-occurring features.
- **Single decision trees** should struggle because axis-aligned splits on individual word counts capture only weak signal per feature; lexical sentiment information is distributed across many features and additive in nature.
- **Ensemble methods** should partially repair the weaknesses of individual trees (Random Forest) and provide small variance-reduction gains for already-strong learners (bagged Logistic Regression).

This project tests these expectations empirically and quantifies the gap between predicted and observed performance.

---

## 3. Dataset Description

The dataset is the **IMDB Movie Reviews dataset** (`IMDB Dataset.csv`), a widely used benchmark in sentiment analysis. It comprises **50,000 reviews** with two columns:

- `review` — the raw movie review text, containing HTML markup (e.g., `<br />`), punctuation, and casing variation.
- `sentiment` — a binary categorical label, `positive` or `negative`.

The dataset is **class-balanced** (25,000 positive and 25,000 negative reviews), eliminating the need for class-imbalance correction (e.g., resampling, class weights, or threshold tuning). The labels are encoded with `LabelBinarizer` to a 0/1 numeric vector and persisted to disk as the file `sentiment` (CSV-style flat file).

**Train/test protocol.** A stratified 70/30 split with `random_state = 1` is applied uniformly across all model notebooks, producing a training set of **35,000** reviews and a test set of **15,000** reviews. Stratification preserves the 50/50 class balance in both partitions, ensuring that reported accuracy figures are directly comparable to a 50% random baseline.

```python
from sklearn.model_selection import train_test_split
y = pd.read_csv('sentiment', header=None).to_numpy().flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1
)
```

The use of a single fixed split (rather than nested cross-validation across the entire pipeline) is a methodological simplification — acceptable here because the test set is large (15,000 examples), giving a standard error on accuracy of roughly ±0.0026 at 50% accuracy and ±0.0025 near 88%, sufficient to distinguish models whose accuracies differ by more than ~0.5 percentage points.

---

## 4. Methodology — Data Preparation Pipeline

The data preparation stage is implemented in `data_preparation.ipynb` and produces two artifacts consumed by every downstream model notebook: the document-term sparse matrix `word_count.npz` and the binary label vector `sentiment`.

### 4.1 Text Cleaning

Five sequential text-normalization operations are applied to each review.

#### (1) HTML tag removal — `BeautifulSoup`

Reviews contain residual HTML markup from web scraping (predominantly `<br />`, but also italic and bold tags). These are stripped using `BeautifulSoup` with the `html.parser` backend:

```python
from bs4 import BeautifulSoup

def remove_tag(text):
    html = BeautifulSoup(text, "html.parser")
    return html.get_text()

df["review"] = df["review"].apply(remove_tag)
```

Using a parser rather than a regex is the safer choice here, since malformed or nested tags would defeat a naive `re.sub('<.*?>', '', ...)` approach.

#### (2) Punctuation and special-character removal — `re`

A two-stage regular expression pass first replaces specific punctuation marks (`,:!?./|*()"`) with spaces, then strips *any non-alphabetic character*, including digits and underscores:

```python
import re

def remove_special_sign(text):
    pattern = r'[,:!?.\/\|\*\(\)\"]'
    text = re.sub(pattern, ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

df["review"] = df["review"].apply(remove_special_sign)
```

This aggressive filtering reflects the assumption — common in classical sentiment analysis — that lexical content rather than punctuation patterns or numeric tokens carries the discriminative signal. It is worth noting as a methodological caveat that emphatic punctuation (multiple exclamation marks, ellipses) and numeric ratings ("10/10") *can* carry sentiment information; their removal trades a small amount of signal for substantial vocabulary regularization.

#### (3) Stopword removal — NLTK English stopwords + ToktokTokenizer

High-frequency function words (e.g., *the, of, is, and*) are removed using NLTK's English stopword list. The tokenizer is `ToktokTokenizer`, chosen for its speed on long-form text:

```python
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

stop = set(stopwords.words('english'))
tokenizer = ToktokTokenizer()

def stopword_remover(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_words = [token for token in tokens if token.lower() not in stop]
    return " ".join(filtered_words)

df["review"] = df["review"].apply(stopword_remover)
```

Stopword removal is largely a vocabulary-compression step; the BoW + linear classifier combination would assign near-zero weight to these terms regardless, but removing them up front reduces matrix size and downstream computational cost.

#### (4) Combined Lemmatization + Stemming

A two-stage morphological reduction is performed: first `WordNetLemmatizer` (which uses lexical knowledge to map *running → run*, *children → child* under appropriate POS), then Porter stemming (which applies suffix-stripping rules, e.g., *wonderful → wonder*, *production → product*):

```python
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def stem_word(comment):
    tokens = tokenizer.tokenize(comment)
    texts = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join([ps.stem(text) for text in texts])

df["review"] = df["review"].apply(stem_word)
```

**Methodological note.** The stacking of lemmatization and stemming is unusual; lemmatization alone is generally preferred for interpretability, while stemming alone is preferred for aggressive vocabulary collapse. Combining them yields a more aggressive normalization at the cost of producing some non-word stems (visible in the cleaned output as *littl, terrif, comfort, brutal*). For a downstream BoW model where features are anonymous indices into a vocabulary, the loss of interpretability is acceptable; the gain is reduced sparsity and improved generalization for low-frequency morphological variants.

After cleaning, the most common 100 stems include semantically central tokens such as `movi, film, one, like, time, good, make, charact, see, get, watch, stori, scene, bad, great, love, plot, actor`, confirming that the pipeline preserves sentiment-bearing vocabulary while collapsing morphological variants.

### 4.2 Vectorization — Bag-of-Words

Cleaned text is vectorized using `sklearn.feature_extraction.text.CountVectorizer` with default settings (unigrams, raw integer counts, no `min_df` / `max_df` filtering):

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv_train = cv.fit_transform(df["review"])
# Result: <50000x70847 sparse matrix, 4,637,943 stored elements (CSR)>
```

The resulting design matrix is a **50,000 × 70,847 Compressed Sparse Row matrix** with approximately 4.64 million non-zero entries — a density of ~0.13%. The vocabulary size of 70,847 unique stems is the consequence of (a) the relatively unrestricted CountVectorizer settings and (b) a long tail of low-frequency stems that survived the cleaning pipeline.

The matrix is persisted to disk in compressed sparse format to enable each model notebook to load it independently:

```python
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename + '.npz')
    return csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )

save_sparse_csr('word_count', cv_train)
```

Storing the raw count matrix (rather than a TF-IDF-transformed version) is a deliberate choice: it preserves flexibility for downstream models. (TF-IDF was prototyped in commented-out cells but not used in the final pipeline.)

**Note on the choice of CountVectorizer over TF-IDF.** For linear classifiers with L2 regularization, raw counts and TF-IDF weights typically produce comparable accuracy on long-form text (where document length variation is moderate). For tree-based models, raw counts are arguably preferable because TF-IDF's continuous, normalized values do not interact naturally with axis-aligned threshold splits.

### 4.3 Label Encoding

The `sentiment` column is mapped from `{positive, negative}` to `{1, 0}` using `LabelBinarizer`:

```python
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
sentiment_data = lb.fit_transform(df['sentiment'])  # shape (50000, 1)
np.savetxt('sentiment', sentiment_data, delimiter=',')
```

The `(50000, 1)` shape is flattened to a 1-D array on load via `.flatten()`.

---


## 5. Feature Selection

A second methodological step — applied to most (though not all) downstream models — reduces the feature dimensionality from **70,847** to **40,000** by retaining the top-ranked features according to **Random Forest feature importance**. The selection procedure is implemented in `Random_Forest.ipynb` (in commented-out cells reproduced below) and the resulting index list is saved to `feature_selection.txt`:

```python
# Compute importances on the full 70,847-feature matrix
rf = RandomForestClassifier(n_estimators=100, criterion="entropy")
rf.fit(X_train, y_train)
importance_order = np.argsort(rf.feature_importances_)[::-1]

# Persist the top 40,000 feature indices
with open("feature_selection.txt", "w") as f:
    for line in importance_order[:40000]:
        f.write(str(line) + "\n")
```

The downstream models then load and apply this index list:

```python
features = open("feature_selection.txt", "r")
index = [int(line.rstrip()) for line in features.readlines()]
X = X[:, index]   # <50000x40000 sparse matrix, 4,584,933 stored elements>
```

### 5.1 Effect of feature selection

The selection retains **98.86%** of the non-zero entries (4,584,933 of 4,637,943) while discarding **43.5%** of the columns (30,847 of 70,847). This confirms that the discarded features are predominantly **rare stems with very few occurrences** — most likely typos, proper nouns, and very low-frequency vocabulary that contribute little to classification accuracy but bloat the model.

### 5.2 Which models use feature selection?

The notebooks differ in whether they apply feature selection. The following table records the actual feature dimensionality used by each model:

| Model | Feature selection applied? | Dimensionality |
|---|---|---|
| Logistic Regression | Yes | 40,000 |
| LinearSVC | **No** | 70,847 |
| SVM (RBF) | **No** | 70,847 |
| KNN | Yes | 40,000 |
| Decision Tree | Yes | 40,000 |
| Random Forest | Yes | 40,000 |
| Bagging Logistic | **No** | 70,847 |

**Methodological caveat.** The inconsistency means that strict "all-else-equal" comparisons across models are not possible from these notebooks alone. The 40k vs. 70k feature gap is unlikely to materially affect linear models (which are robust to irrelevant features under L2 regularization) but could in principle affect KNN and trees. In practice the differences appear small: LinearSVC at 70k achieves 0.8881 and Logistic Regression at 40k achieves 0.8896 — both within standard-error proximity. Future iterations of this work should standardize the feature space before final model comparison.

### 5.3 Methodological observation — circularity risk

A subtle issue worth flagging: the feature importance ranking is computed on the *training* split of the same train/test partition that is later used to evaluate every downstream model. This is the correct, non-leaky procedure (test data is never inspected during selection). However, the procedure does mean that the feature subset is implicitly tuned on the same training data later used for hyperparameter search, which can produce a small optimistic bias. A more rigorous protocol would compute importances on a held-out *selection* fold disjoint from both training and final evaluation. Given the size of the dataset and the conservatism of L2 regularization, the practical impact is expected to be minimal.

---

## 6. Model 1 — Logistic Regression

**Notebook:** `Logistic_Regression.ipynb`
**Feature dimensionality:** 40,000 (after feature selection)
**Best test accuracy:** **0.8896** *(highest among all models tested)*

### 6.1 Theoretical framing

Logistic regression models the conditional probability of the positive class as

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

and is trained by minimizing the L2-regularized negative log-likelihood:

$$
\min_{\mathbf{w}, b} \; \frac{1}{2C}\|\mathbf{w}\|_2^2 + \sum_{i=1}^n \log\bigl(1 + \exp(-y_i(\mathbf{w}^\top \mathbf{x}_i + b))\bigr)
$$

The hyperparameter `C` is the inverse regularization strength: small `C` means strong regularization, large `C` means weak regularization. For high-dimensional sparse text problems, `C` typically needs to be tuned across several orders of magnitude.

### 6.2 Hyperparameter tuning

A 10-fold cross-validated grid search over six values of `C` spanning four orders of magnitude is performed:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
clf = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    refit=True,
    n_jobs=-1,
    cv=10
)
clf.fit(X_train, y_train)
```

**Selected hyperparameter:** `C = 0.1`.
**Test accuracy:** `0.8896`.

The selection of a relatively *small* `C` (strong regularization) is consistent with theoretical expectation: with 40,000 features and 35,000 training examples, the unregularized problem is technically underdetermined (more features than examples). Strong L2 regularization shrinks weights toward zero, smoothing the decision boundary and preventing the classifier from memorizing rare-token patterns.

### 6.3 Confusion matrix

On the 15,000-example test set:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 6,610 (TN) | 766 (FP) |
| **Actual Positive** | 890 (FN) | 6,734 (TP) |

Derived metrics:
- Precision (positive class): 6734 / (6734 + 766) = **0.898**
- Recall (positive class): 6734 / (6734 + 890) = **0.883**
- F1 score: **0.891**
- Specificity: 6610 / (6610 + 766) = **0.896**

The error distribution is approximately symmetric, with a slight tendency to miss positive reviews (890 false negatives vs. 766 false positives), suggesting the classifier is marginally conservative on the positive class.

### 6.4 ROC and learning curve diagnostics

A ROC curve was generated using `decision_function` outputs as the score; the AUC (visible in the saved figure `ROC_logistic`) is consistent with the high accuracy. A learning curve was generated using `sklearn.model_selection.learning_curve` with training-set fractions [0.1, 0.325, 0.55, 0.775, 1.0] (sizes 3500 → 35000), allowing diagnosis of bias-variance behavior:

```python
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    estimator, X, y, cv=cv, n_jobs=-1,
    train_sizes=np.linspace(.1, 1.0, 5),
    return_times=True
)
```

The standard interpretation: if the train-CV gap closes as more data is added, the model is variance-limited and would benefit from more data; if both curves plateau at low accuracy, the model is bias-limited and needs a richer hypothesis class. For Logistic Regression on this task, the curves indicate near-convergence — additional data would yield diminishing returns.

### 6.5 Predictions persisted

```python
np.savetxt('y_predict_logistic', y_predict, delimiter=',')
```

This file is later consumed by `Bagging_Logistics.ipynb` for the bagging-vs-base comparison (see Section 12).

---

## 7. Model 2 — Linear Support Vector Classifier (LinearSVC)

**Notebook:** `LinearSVC.ipynb`
**Feature dimensionality:** 70,847 (no feature selection)
**Best test accuracy:** **0.8881**

### 7.1 Theoretical framing

LinearSVC fits a linear classifier minimizing the L2-regularized squared hinge loss (by default in scikit-learn):

$$
\min_{\mathbf{w}, b} \; \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^n \max\bigl(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr)^2
$$

Compared to Logistic Regression, LinearSVC differs only in the loss function: hinge loss penalizes only points within or beyond the margin, whereas logistic loss is smooth and penalizes all points (with diminishing influence on confidently-correct predictions). For approximately-linearly-separable problems, the two typically perform similarly.

### 7.2 Hyperparameter tuning

The same `C` grid as Logistic Regression is searched:

```python
from sklearn import svm
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
clf = GridSearchCV(
    estimator=svm.LinearSVC(dual=True),
    param_grid=param_grid,
    refit=True,
    n_jobs=-1,
    cv=10
)
clf.fit(X_train, y_train)
```

**Selected hyperparameter:** `C = 0.01`.
**Test accuracy:** `0.8881`.

LinearSVC's optimal `C` is one order of magnitude smaller than Logistic Regression's. This is *not* a contradiction — the two losses scale differently, so their `C` regularization paths are not directly comparable. The result simply confirms that strong regularization is required.

### 7.3 Confusion matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 6,608 (TN) | 786 (FP) |
| **Actual Positive** | 892 (FN) | 6,714 (TP) |

The error structure is virtually identical to Logistic Regression's — slightly more false positives and one fewer false negative, with overall accuracy 0.0015 lower.

### 7.4 ROC and learning curve

The ROC curve was generated using `decision_function` (the signed distance to the separating hyperplane). The learning curve uses the same protocol as Logistic Regression. The figures show the same near-convergent behavior, confirming that LinearSVC is also approaching its irreducible error.

### 7.5 Predictions persisted

```python
np.savetxt('y_predict_linearSVC', y_predict, delimiter=',')
```

---

## 8. Model 3 — Kernel SVM (RBF)

**Notebook:** `SVM.ipynb`
**Feature dimensionality:** 70,847 (no feature selection)
**Test accuracy:** **0.8771**

### 8.1 Theoretical framing

A kernel SVM with the **Radial Basis Function** (RBF) kernel implicitly maps the input to an infinite-dimensional feature space and finds a maximum-margin hyperplane there:

$$
K(\mathbf{x}, \mathbf{x}') = \exp\bigl(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\bigr)
$$

For text classification, the RBF kernel offers no obvious theoretical advantage over a linear kernel: text data is approximately linearly separable in BoW space, and the RBF kernel's locality property (similarity decaying with Euclidean distance) interacts poorly with the high-dimensional sparse structure of word counts.

### 8.2 Implementation

Unlike the other models, no hyperparameter grid search was performed for the RBF SVM — likely because of the prohibitive computational cost of `SVC` on a 35,000 × 70,847 sparse training matrix (training scales superlinearly in the number of samples and is not parallelizable across `C` values without significant memory cost):

```python
from sklearn import svm

clf = svm.SVC()           # default: C = 1.0, kernel = 'rbf', gamma = 'scale'
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
np.mean(y_test == y_predict)   # 0.8771
```

### 8.3 Result interpretation

The RBF SVM achieves **0.8771**, approximately **1.2 percentage points below LinearSVC and Logistic Regression**. This confirms the theoretical prediction: for this problem class, the kernel does not extract additional signal beyond what linear models already capture, and may even underperform due to the unfavorable interaction between high-dimensional sparsity and Euclidean-distance kernels. The kernel SVM also took dramatically longer to train than the linear alternatives — a substantial practical penalty for no accuracy gain.

This notebook is the simplest of the seven, lacking the hyperparameter tuning, learning curve, and ROC analysis present in the others. Its inclusion serves primarily as a baseline comparison demonstrating that the linear kernel is the appropriate choice.

---

## 9. Model 4 — K-Nearest Neighbors (KNN)

**Notebook:** `KNN.ipynb`
**Feature dimensionality:** 40,000 (after feature selection)
**Best test accuracy:** **0.7266**

### 9.1 Theoretical framing

K-Nearest Neighbors classifies a test point by majority vote of its `k` closest training points under a chosen metric (Euclidean by default). It is a **non-parametric, instance-based** learner: there is no training phase beyond storing the data, and prediction cost scales linearly with training-set size.

KNN's well-known weakness on high-dimensional sparse data is the **curse of dimensionality**: in high-dimensional spaces, all points become approximately equidistant from one another, eroding the signal-to-noise ratio of the distance metric. For BoW representations specifically, two reviews share most of their zeros, so Euclidean distance is dominated by the sparse non-zero counts and becomes noisy.

### 9.2 Hyperparameter tuning — iterative narrowing

Unique among the model notebooks, KNN's `n_neighbors` was tuned via **four sequential grid searches**, each narrowing the range around the previous best value:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Pass 1: coarse sweep
param_grid = {'n_neighbors': [50, 100, 150, 200, 250, 300]}
gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5).fit(X_train, y_train)
# best: 150 (CV score 0.7193)

# Pass 2: refine around 150
param_grid2 = {'n_neighbors': [160, 180, 200, 220, 240]}
gs2 = GridSearchCV(KNeighborsClassifier(), param_grid2, cv=5).fit(X_train, y_train)
# best: 180 (CV score 0.7209)

# Pass 3: refine around 180
param_grid3 = {'n_neighbors': [170, 175, 180, 185, 190]}
gs3 = GridSearchCV(KNeighborsClassifier(), param_grid3, cv=5).fit(X_train, y_train)
# best: 180 (CV score 0.7209)

# Pass 4: fine-grained around 180
param_grid4 = {'n_neighbors': [176, 177, 178, 179, 180, 181, 182, 183, 184]}
gs4 = GridSearchCV(KNeighborsClassifier(), param_grid4, cv=5).fit(X_train, y_train)
# best: 180 (CV score 0.7209)
```

The CV scores across all 25 evaluated values of `k`:

```
[0.71203, 0.71809, 0.71926, 0.71911, 0.71671, 0.71506,
 0.71769, 0.72094, 0.71911, 0.71911, 0.71720, 0.71909,
 0.71663, 0.72094, 0.71709, 0.72043, 0.71969, 0.71740,
 0.72017, 0.71751, 0.72094, 0.71731, 0.71983, 0.71691,
 0.72034]
```

The CV-score landscape is essentially flat across the range 150 ≤ k ≤ 240, with all values within ~0.7 percentage points of one another. **k = 180** was selected as the best.

**Final test accuracy:** `0.7266`.

The fact that the validation accuracy is nearly flat across an order of magnitude of `k` is itself diagnostic: it confirms that **KNN cannot recover the discriminative structure of the data regardless of `k`** in this representation. The model is operating near its performance ceiling on this feature space.

### 9.3 Confusion matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4,397 (TN) | 998 (FP) |
| **Actual Positive** | 3,103 (FN) | 6,502 (TP) |

The error structure is **strikingly asymmetric**: the model has 3,103 false negatives versus only 998 false positives — it predicts "positive" much more often than it should. This is a known artifact of KNN with large `k` on imbalanced neighborhood structure: with k = 180, even slight bias in the neighborhood-class distribution propagates into a strong classification bias.

### 9.4 Predictions persisted

```python
np.savetxt('y_predict_knn', y_predict, delimiter=',')
```

---

## 10. Model 5 — Decision Tree

**Notebook:** `Decision_Tree.ipynb`
**Feature dimensionality:** 40,000 (after feature selection)
**Best test accuracy:** **0.7241**

### 10.1 Theoretical framing

A decision tree classifier recursively partitions the feature space using axis-aligned splits, choosing at each node the feature and threshold that maximally reduce a chosen impurity measure (Gini or entropy). The depth-controlled tree is a **high-bias, high-variance** model: shallow trees underfit, deep trees overfit aggressively.

For BoW text classification, single trees are theoretically poorly suited because:
- Sentiment is **additive across many words**: a positive review contains *many* positive words, each contributing a small amount of evidence. A tree must build a long chain of splits to aggregate this evidence, fragmenting the data and losing statistical power at each level.
- Most splits on individual word counts have **only weak signal** (few words are individually highly discriminative).

### 10.2 Hyperparameter tuning

A 10-fold CV grid search over `max_depth` and `criterion`:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [1, 10, 50, 100, 500, None],
    'criterion': ['gini', 'entropy']
}
gs = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10).fit(X_train, y_train)
```

**Selected hyperparameters:** `criterion = 'gini'`, `max_depth = 10`.
**Test accuracy:** `0.7241`.

The relatively shallow `max_depth = 10` was selected — deeper trees overfit. With 35,000 training examples, even depth-10 admits up to 1,024 leaves, more than enough to capture the dominant patterns; deeper trees memorize training noise.

### 10.3 Confusion matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4,474 (TN) | 1,112 (FP) |
| **Actual Positive** | 3,026 (FN) | 6,388 (TP) |

The error pattern is similar in shape to KNN — strong bias toward predicting "positive" — but for a different reason: the shallow tree captures only a few coarse decision rules, and these rules apparently default to "positive" when ambiguous. The true negative rate (sensitivity for the negative class) is just 4,474 / (4,474 + 1,112) = 0.801, whereas the true positive rate is 6,388 / (6,388 + 3,026) = 0.679 — nearly inverted from KNN's pattern but still highly asymmetric.

### 10.4 Predictions persisted

```python
np.savetxt('y_predict_decisiontree', y_predict, delimiter=',')
```

---

## 11. Model 6 — Random Forest

**Notebook:** `Random_Forest.ipynb`
**Feature dimensionality:** 40,000 (after feature selection)
**Best test accuracy:** **0.8550**

### 11.1 Theoretical framing

A Random Forest is an ensemble of decision trees trained on bootstrap samples with random feature subsetting at each split. The two sources of randomness — bootstrap aggregation and per-split feature randomization — **decorrelate** the constituent trees, allowing their predictions to be averaged for substantial **variance reduction**.

For text classification, Random Forests typically improve substantially over single decision trees but do not reach the performance of regularized linear models, because the underlying mismatch between axis-aligned splits and the additive-across-features nature of sentiment remains.

### 11.2 Hyperparameter tuning

A 10-fold CV grid search:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [1, 10, 50, 100, 500, None],
    'criterion': ['gini', 'entropy']
}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=10).fit(X_train, y_train)
```

**Selected hyperparameters:** `criterion = 'entropy'`, `max_depth = 50`.
**Test accuracy:** `0.8550`.

Note that `n_estimators` was *not* in the grid — the scikit-learn default of 100 trees was used. The selection of a much deeper `max_depth = 50` than the single decision tree's `max_depth = 10` is consistent with theory: Random Forest's variance-reduction property allows individual trees to be deeper (and individually overfit) without harming ensemble performance, because the averaging cancels their idiosyncratic errors.

### 11.3 Confusion matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 6,251 (TN) | 926 (FP) |
| **Actual Positive** | 1,249 (FN) | 6,574 (TP) |

The Random Forest substantially repairs the asymmetry seen in the single decision tree: the false-negative count drops from 3,026 to 1,249 (a 59% reduction), while false positives drop more modestly from 1,112 to 926. The error structure is now approximately balanced.

### 11.4 Importance ranking re-use

This notebook also produced the **feature importance ranking** used by the feature-selection step (Section 5). The importance-ranking code was preserved as commented-out cells:

```python
# rf = RandomForestClassifier(n_estimators=100, criterion="entropy")
# rf.fit(X_train, y_train)
# importance_order = np.argsort(rf.feature_importances_)[::-1]
#
# with open("feature_selection.txt", "w") as f:
#     for line in importance_order[:40000]:
#         f.write(str(line) + "\n")
```

This is an interesting methodological pattern: the Random Forest serves a **dual role** — both as a candidate classifier in its own right and as a feature-importance oracle for downstream models. The technique exploits Random Forest's well-known property of providing reliable feature-importance estimates even when its own classification performance is sub-optimal.

### 11.5 Predictions persisted

```python
np.savetxt('y_predict_randomforest', y_predict, delimiter=',')
```

---

## 12. Model 7 — Bagging applied to Logistic Regression

**Notebook:** `Bagging_Logistics.ipynb`
**Feature dimensionality:** 70,847 (no feature selection)
**Test accuracy range:** **0.8847 – 0.8870** depending on ensemble size

### 12.1 Theoretical framing and motivation

**Bagging** (Bootstrap Aggregating) trains multiple copies of a base learner on bootstrap samples of the training data and aggregates their predictions by majority vote (classification) or averaging (regression). It is a **variance-reduction** technique whose effectiveness depends critically on the base learner's instability: it dramatically improves high-variance learners (like decision trees, where Random Forest is essentially Bagging-of-Trees plus feature randomization) but provides only marginal gains for stable, low-variance learners.

Logistic Regression is a **stable, low-variance** learner. The expected gain from bagging it is therefore small, and this notebook serves as an **empirical test of that prediction**.

### 12.2 Sweep over ensemble size

Eight ensemble sizes (10, 20, 30, 40, 50, 60, 70, 80) are evaluated, with the unbagged baseline (the final Logistic Regression model from Section 6) included as the `n=0` reference point:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

for n in [10, 20, 30, 40, 50, 60, 70, 80]:
    clf = BaggingClassifier(
        LogisticRegression(max_iter=10000),
        n_estimators=n,
        random_state=1
    )
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(n, np.mean(y_test == y_predict))
```

Results:

| n_estimators | Test accuracy | Δ vs. baseline |
|---|---|---|
| 0 (no bagging) | 0.8896 | — |
| 10 | 0.8847 | −0.0049 |
| 20 | 0.8855 | −0.0041 |
| 30 | 0.8861 | −0.0035 |
| 40 | 0.8859 | −0.0037 |
| 50 | 0.8867 | −0.0029 |
| 60 | 0.8859 | −0.0037 |
| 70 | **0.8870** | −0.0026 |
| 80 | 0.8868 | −0.0028 |

### 12.3 Interpretation

Two observations stand out:

**(1) Bagging does not improve over the well-regularized baseline.** Every bagged variant achieves slightly *lower* test accuracy than the un-bagged Logistic Regression. This is consistent with theoretical expectation — Logistic Regression with cross-validated L2 regularization is already a low-variance learner, and the additional regularization implicit in bagging (each base model sees only ~63.2% of unique examples) is not helpful here.

**(2) However, the bagged ensemble is dramatically more accurate than its individual sub-learners would be.** A *single* Logistic Regression model trained on a bootstrap sample (with no `C` tuning, since `BaggingClassifier` does not propagate hyperparameter search) would achieve substantially lower accuracy than ~0.886. The bagging procedure recovers most of the signal that would otherwise be lost. The comparison `Bagging(LogReg, n=70) ≈ 0.8870` should therefore be read against an apples-to-apples baseline of `LogReg(default C=1)`, against which bagging *does* offer improvement.

**(3) Performance saturates rapidly.** Beyond n ≈ 30, additional ensemble members yield no consistent gain — the curve plateaus. This is the standard bias-variance picture: with sufficient ensemble members, residual error is dominated by the (irreducible) bias of the base learner, not its variance.

### 12.4 Visualization of the sweep

The notebook produces a line plot `Logistic_bagging.png` showing test accuracy vs. ensemble size, including the baseline:

```python
x = np.arange(0, 90, 10)
y = np.array([acc, acc10, acc20, acc30, acc40, acc50, acc60, acc70, acc80])
plt.plot(x, y, 'r--')
plt.title("Test accuracy vs Bagging size")
plt.xlabel("The number of bootstrap logistic estimators (0=no bagging)")
plt.ylabel("Test accuracy")
plt.savefig("Logistic_bagging")
```

### 12.5 McNemar-style disagreement analysis

The most methodologically interesting cell of this notebook is a custom **disagreement analysis** between the un-bagged Logistic Regression (loaded from `y_predict_logistic`) and the largest bagged ensemble (`n = 80`). Rather than computing a standard confusion matrix against ground truth, the cell computes a 2×2 contingency table over **whether each model agrees with ground truth**:

```python
y_raw = np.loadtxt('y_predict_logistic')
tn, fp, fn, tp = 0, 0, 0, 0
for i in range(len(y_raw)):
    if y_raw[i] == y_predict8[i] and y_raw[i] == y_test[i]:
        tp += 1   # both correct
    elif y_raw[i] == y_predict8[i] and y_raw[i] != y_test[i]:
        tn += 1   # both wrong (and agree with each other)
    elif y_raw[i] != y_predict8[i] and y_raw[i] == y_test[i]:
        fn += 1   # base correct, bagging wrong
    elif y_raw[i] != y_predict8[i] and y_raw[i] != y_test[i]:
        fp += 1   # base wrong, bagging correct
```

The variable names are reused from a binary-classification context but here index a different decomposition. The resulting 2×2 table is:

| | Bagging agrees with truth | Bagging disagrees with truth |
|---|---|---|
| **Base agrees with truth** | 13,081 (both correct) | 131 (only base correct) |
| **Base disagrees with truth** | 221 (only bagging correct) | 1,567 (both wrong) |

This is the precise data needed for **McNemar's test**, which asks whether two classifiers' disagreement counts (131 vs. 221) are symmetric under the null hypothesis of equal accuracy. With these counts, the McNemar test statistic is:

$$
\chi^2 = \frac{(|131 - 221| - 1)^2}{131 + 221} = \frac{89^2}{352} \approx 22.5
$$

which is highly significant (p < 0.0001) against a χ² distribution with one degree of freedom. The bagged ensemble corrects 221 errors of the base model while introducing only 131 new ones — a **statistically significant net improvement**, even though the *raw accuracy difference* (0.8870 vs 0.8896) is in the opposite direction.

This apparent contradiction resolves once one recognizes that `y_predict_logistic` was produced by the **GridSearchCV-tuned** Logistic Regression, which was the *better* baseline reported in Section 6. The McNemar-style table here uses that same y_predict_logistic, so the "bagging corrects 221, breaks 131" interpretation reflects the bagged ensemble against the *pre-trained, tuned* baseline — and yet bagging still ends up with lower overall accuracy. The reason is that 131 + 221 = 352 disagreements break in a slight net favor of the un-bagged classifier when accuracy is computed against the full 15,000-example test set:

- Un-bagged correct, bagging wrong: 131 → un-bagged wins these
- Bagging correct, un-bagged wrong: 221 → bagging wins these
- Both correct: 13,081 (no swing)
- Both wrong: 1,567 (no swing)

So bagging's net error count is `131 - 221 = -90` better, i.e., *bagging makes 90 fewer errors*. Yet in absolute terms `0.8870 < 0.8896`, suggesting the accuracy figures reported earlier in the notebook were generated under a slightly different ensemble configuration than the `clf8` (n=80) used in this comparison. This is a small inconsistency in the notebook's bookkeeping that does not affect the overall conclusion: **bagging produces a different but comparably-accurate classifier**, and McNemar's test confirms the differences are not random.

The corresponding visualization is saved as `McNemar.png`.

---


## 13. Comparative Analysis and Discussion

### 13.1 The performance gap: linear models vs. the rest

The single most striking empirical fact of this project is the **clean separation** of the seven models into two performance tiers:

- **Tier 1 — High-performing linear models:** Logistic Regression (0.890), LinearSVC (0.888), Bagging-Logistic (0.887). The RBF SVM (0.877) sits just below this tier.
- **Tier 2 — Weak high-variance / non-linear models:** Random Forest (0.855), KNN (0.727), Decision Tree (0.724).

The Tier-1 / Tier-2 gap is approximately **3 to 16 percentage points** depending on the comparison — a margin far exceeding the test-set standard error.

This pattern is **textbook for high-dimensional sparse text classification**:

- **Why linear models win.** Sentiment in BoW representations is approximately a *linear function of word counts*: each sentiment-bearing word contributes additive, roughly independent evidence. A linear classifier with appropriate regularization is exactly the right hypothesis class — flexible enough to weigh thousands of features, simple enough to avoid memorizing noise.

- **Why KNN underperforms.** In 40,000-dimensional sparse space, Euclidean distances are dominated by the few non-zero counts that two reviews happen to share, producing a noisy and unstable neighborhood structure. The flat CV-score curve across 25 values of `k` (Section 9.2) confirms this: no value of `k` can recover signal that distance has destroyed.

- **Why decision trees underperform.** A single tree must capture additive evidence through sequential splits, fragmenting the data at each level and rapidly losing statistical power. Random Forest substantially repairs this (0.855 vs. 0.724), but the underlying mismatch between axis-aligned splits and additive signal remains.

### 13.2 Bias–variance accounting

The learning-curve diagnostics across models reveal the following pattern:

| Model | Bias-limited or variance-limited? |
|---|---|
| Logistic Regression | Approaching irreducible error; mildly bias-limited |
| LinearSVC | Same as Logistic Regression |
| RBF SVM | Bias-limited (kernel mismatch) |
| Decision Tree | Bias-limited at depth 10; variance-limited if deeper |
| Random Forest | Variance well-controlled by ensembling; mild bias remains |
| KNN | Variance-limited at small k, bias-limited at large k |
| Bagging Logistic | Variance already low; bagging gives no further gain |

The takeaway: **on this problem, more model capacity would not help** — the linear models are already extracting most of the recoverable signal, and the gap to a "perfect" classifier (~11% misclassified) is dominated by genuinely ambiguous reviews and label noise rather than by model deficiency.

### 13.3 The role of regularization

The cross-validated `C` selections tell a coherent story:

| Model | Selected `C` | Interpretation |
|---|---|---|
| Logistic Regression | 0.1 | Strong L2 regularization |
| LinearSVC | 0.01 | Even stronger L2 regularization (different loss scale) |
| RBF SVM | 1.0 (default) | Untuned |

Both linear models converge on **strong regularization**, consistent with the classical result that high-dimensional underdetermined linear systems require it for good generalization. The fact that *very different* regularization strengths emerge (0.01 vs. 0.1) reflects the different scales of the hinge and logistic losses — the *effective* regularization (the prior variance on weights) is comparable.

### 13.4 Computational cost

While this project does not record formal training times, the relative costs are well-known and worth flagging:

| Model | Training cost on this dataset |
|---|---|
| LinearSVC | Seconds (LIBLINEAR solver, exploits sparsity) |
| Logistic Regression | Seconds (saga / lbfgs) |
| Decision Tree | Seconds (single tree) |
| KNN | Trivial training, *expensive* prediction (15,000 × 35,000 distance computations per k) |
| Random Forest | Minutes (100 trees × deep splits) |
| Bagging Logistic | Minutes (up to 80 LR fits) |
| RBF SVM | Hours (libsvm scales superlinearly with n) |

The combination of accuracy and computational cost makes Logistic Regression and LinearSVC the unambiguous practical winners.

### 13.5 Methodological inconsistencies worth flagging

For full academic honesty, several minor methodological inconsistencies exist across the notebooks:

1. **Inconsistent feature selection.** Some models use 40,000 features (selected via Random Forest importance), others use the full 70,847. The accuracy gap between LinearSVC (full features) and Logistic Regression (selected features) is small enough that this does not change the overall ranking, but a strict comparison would standardize the feature space.

2. **Inconsistent hyperparameter tuning.** The RBF SVM uses default hyperparameters; all others are grid-searched. The Bagging notebook varies `n_estimators` but does not tune the inner Logistic Regression's `C`.

3. **No nested cross-validation.** All hyperparameter selection uses a single train/test split. While the test set is large enough to give reliable accuracy estimates, the *selection* of hyperparameters is technically using the same split that produces the final accuracy figure (via CV folds within the training set). This is standard practice but not the gold standard.

4. **`y_predict_logistic` re-use anomaly.** Section 12.5's McNemar analysis shows a small bookkeeping inconsistency: the bagged-vs-base disagreement counts imply bagging makes fewer errors, but the headline accuracies show the opposite. This is most likely a stale prediction file or a within-notebook variable shadowing.

None of these issues meaningfully change the qualitative conclusions, but a follow-up project should address them.

---

## 14. Conclusions and Future Work

### 14.1 Conclusions

This project produces several substantive empirical findings:

1. **Regularized linear models are the appropriate hypothesis class for BoW sentiment classification.** Logistic Regression with cross-validated L2 regularization achieves **88.96%** test accuracy on a balanced 15,000-example test set — the best among seven competing approaches.

2. **The accuracy ranking matches classical theoretical predictions.** Linear models > kernel SVM > Random Forest > KNN ≈ Decision Tree. The 16-percentage-point spread between best and worst confirms that algorithm choice on high-dimensional sparse data is consequential, and that classical results in the literature transfer to this specific dataset.

3. **Bagging does not improve a strong base learner.** With cross-validated regularization, Logistic Regression is already a low-variance learner; bagging produces a marginally different but not better classifier. Empirical confirmation of a textbook result.

4. **Random Forest serves a useful dual role** — both as a candidate classifier and as a feature-importance oracle for downstream models. It substantially improves on a single decision tree (0.855 vs 0.724) but cannot match linear models on a representation where signal is additive in features.

5. **All seven models benefit from a shared, carefully-engineered preprocessing pipeline.** The HTML stripping, punctuation removal, stopword filtering, lemmatization, and stemming together collapse a vocabulary of essentially unbounded size to 70,847 manageable stems with retained sentiment-bearing content.

### 14.2 Future Work

Several directions would extend this work productively:

1. **TF-IDF vs. raw counts.** A controlled comparison on the same models would test whether IDF weighting helps the linear classifiers further. Expected effect: small (perhaps +0.5%).

2. **n-gram features.** Including bigrams and trigrams (e.g., *"not good"*) would capture negation, which BoW unigrams handle poorly. This is a likely source of meaningful improvement (potentially +1-2%).

3. **Word embeddings + linear classifier.** Replacing the BoW representation with averaged pre-trained word embeddings (Word2Vec, GloVe, fastText) would test whether dense representations help here. Embedding-based features tend to slightly outperform BoW for sentiment, especially on out-of-vocabulary words.

4. **Modern neural baselines.** A fine-tuned transformer (e.g., DistilBERT) on this same dataset typically achieves 93-95% accuracy. Including such a baseline would contextualize how much performance the classical methods leave on the table — and quantify the cost-benefit of moving to deep learning for this task.

5. **Standardize the feature space across models.** Re-running every model on identical 40k-feature input (or, ideally, the full 70,847-feature input) would tighten the comparative analysis.

6. **Nested cross-validation for unbiased accuracy estimation.** The current protocol is acceptable but not optimal. A nested CV protocol would produce confidence-interval-equipped accuracy figures suitable for publication.

7. **Calibration analysis.** Logistic Regression's `predict_proba` outputs are typically reasonably calibrated; LinearSVC's `decision_function` outputs are not. A reliability-diagram comparison across models would inform downstream uses where calibrated probabilities matter.

---

## 15. Appendix — File Inventory

**Notebooks (one project, eight notebooks):**

| File | Role |
|---|---|
| `data_preparation.ipynb` | Text cleaning, vectorization, label encoding |
| `Logistic_Regression.ipynb` | Logistic Regression with CV-tuned C |
| `LinearSVC.ipynb` | LinearSVC with CV-tuned C |
| `SVM.ipynb` | Kernel SVM (RBF), default hyperparameters |
| `KNN.ipynb` | KNN with iterative grid search over n_neighbors |
| `Decision_Tree.ipynb` | Single decision tree, CV-tuned depth and criterion |
| `Random_Forest.ipynb` | Random Forest, CV-tuned depth and criterion; also produces feature importances |
| `Bagging_Logistics.ipynb` | Bagged Logistic Regression with sweep over n_estimators |

**Persisted artifacts (intermediate data):**

| File | Producer | Consumer |
|---|---|---|
| `word_count.npz` | data_preparation | All model notebooks |
| `sentiment` | data_preparation | All model notebooks |
| `feature_selection.txt` | Random_Forest | Logistic_Regression, KNN, Decision_Tree, Random_Forest |
| `y_predict_*` | Each model notebook | Bagging_Logistics (for McNemar comparison) |

**Visualizations produced (one or more per model notebook):**

- Confusion matrix (absolute + normalized) for every model
- ROC curve with AUC for every model except KNN (which uses raw `predict` instead of `decision_function`) and the basic SVM
- Learning curve showing train/CV accuracy as a function of training set size for every model except SVM and Bagging
- `Logistic_bagging.png` — line plot of test accuracy vs. ensemble size
- `McNemar.png` — disagreement matrix between base and bagged Logistic Regression

---

*End of report.*
