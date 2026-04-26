# Tweets Political Ideology Classification — Democrat vs. Republican

**Project Report — STAT 451 Final Project**

**Authors of original work:** Han Cao, Siyi He, Qiwen Zeng
**Course:** STAT 451 (Introduction to Machine Learning), University of Wisconsin–Madison
**Repository:** PoliIdeaClassify
**Final test accuracy:** 76% (Multinomial Naive Bayes)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Motivation](#2-project-goals-and-motivation)
3. [Dataset](#3-dataset)
4. [Methodology — Text Preprocessing Pipeline](#4-methodology--text-preprocessing-pipeline)
5. [Vectorization — TF-IDF on Tweet-Aware Tokens](#5-vectorization--tf-idf-on-tweet-aware-tokens)
6. [Model 1 — Multinomial Naive Bayes](#6-model-1--multinomial-naive-bayes)
7. [Model 2 — Linear Support Vector Machine](#7-model-2--linear-support-vector-machine)
8. [Model 3 — XGBoost](#8-model-3--xgboost)
9. [Model Comparison and Selection](#9-model-comparison-and-selection)
10. [McNemar's Test — Are the Models Statistically Different?](#10-mcnemars-test--are-the-models-statistically-different)
11. [Comparative Analysis and Discussion](#11-comparative-analysis-and-discussion)
12. [Conclusions and Future Work](#12-conclusions-and-future-work)
13. [Appendix — File Inventory and Hyperparameter Tables](#13-appendix--file-inventory-and-hyperparameter-tables)

---

## 1. Executive Summary

This project addresses **binary text classification of political ideology** — given the body of a tweet, predict whether its author is a Democrat or a Republican. The dataset is a publicly available Kaggle corpus of **84,502 tweets** drawn from 433 U.S. politicians' accounts (~200 tweets per user), with class labels nearly balanced (50.2% Republican / 49.8% Democrat).

The team builds a complete classification pipeline:

1. **Text preprocessing**: handle removal, punctuation/digit stripping, casual-tweet-aware tokenization, English stopword removal (plus the Twitter-specific stopword "RT"), and Porter stemming.
2. **Feature extraction**: TF-IDF vectorization using NLTK's `TweetTokenizer` (which handles hashtags, @-mentions, and emoji-style tokens).
3. **Model comparison**: Multinomial Naive Bayes, Linear Support Vector Machine, and XGBoost — each tuned by grid or random search with 4-fold cross-validation.
4. **Statistical comparison**: a McNemar's test between the top two models (Naive Bayes and SVM) to assess whether their differences are statistically meaningful.
5. **Multi-metric evaluation**: accuracy, precision, recall, F1, and Matthews correlation coefficient (MCC).

**Headline results on a 17,292-tweet stratified test set:**

| Model | Test Accuracy | Test F1 | Test MCC |
|---|---|---|---|
| **Multinomial Naive Bayes** | **0.757** | **0.771** | **0.514** |
| Linear SVM (C = 1) | 0.753 | 0.765 | 0.506 |
| XGBoost (tuned) | 0.701 | — | — |

**Three substantive findings emerge:**

1. **Multinomial Naive Bayes is the best-performing model**, achieving 76% test accuracy and the highest scores across every metric measured. Despite its simplicity and naive independence assumption, it slightly outperforms the linear SVM and substantially outperforms XGBoost on this high-dimensional sparse text representation.

2. **The Naive Bayes / SVM gap is small but statistically real.** A McNemar's test between the two models yields *p* = 0.001, rejecting the null hypothesis of equal error rates. This is a genuinely useful methodological inclusion — accuracy differences of less than 1 percentage point are easy to dismiss as noise without an explicit significance test.

3. **XGBoost underperforms substantially** (~5 percentage points lower than the two linear methods). This is consistent with the well-known finding that gradient-boosted trees do not exploit the additive, sparse signal structure of bag-of-words text representations as effectively as naive linear models do.

The project's central methodological contribution is its careful adherence to a clean preprocessing pipeline (handle removal *first*, before stopword filtering interacts with `@` symbols), its use of `TweetTokenizer` (instead of a generic regex tokenizer that would mishandle hashtags), and its inclusion of statistical model comparison rather than relying on raw accuracy.

---

## 2. Project Goals and Motivation

### 2.1 Context — Social Media and Political Ideology

The team's framing situates the project in a current-events context: the 2020 U.S. presidential election, against a backdrop of public concern about social media's role in shaping (or distorting) political discourse. Citing reporting on platforms' potential to manipulate political ideologies, the team motivates the project as a step toward **automatic monitoring of political opinion on social media**.

The specific question they pose:

> Given a tweet's text, can we accurately predict whether its author is a Democrat or a Republican?

This is a well-defined binary classification problem with a natural large-scale data source (Twitter) and clear downstream applications.

### 2.2 Project Objectives

1. **Build a complete text-classification pipeline** for tweets — from raw social-media-formatted text to a deployable predictive model.
2. **Compare three model families** with different inductive biases — a generative probabilistic model (Naive Bayes), a discriminative max-margin model (SVM), and a flexible nonlinear ensemble (XGBoost) — on the same processed feature representation.
3. **Tune each model with cross-validated hyperparameter search** to ensure the comparison is fair.
4. **Apply formal statistical model comparison** (McNemar's test) to determine whether the top-ranked models are meaningfully different.
5. **Assess models with multiple metrics** (accuracy, precision, recall, F1, MCC) to characterize their behavior beyond a single summary number.

### 2.3 Stated Applications

The team identifies three downstream applications for a working classifier:

1. **Public-opinion monitoring** — sample tweets at scale and track shifts in partisan distribution over time, potentially correlated with key political events.
2. **Political-bot identification** — automated accounts that post stylistically uniform political content can be detected as outliers from typical human-tweet distributions.
3. **Demographic mapping of political ideology** — combine ideology classification with user metadata to produce geographic/demographic distributions of partisan content.

These are reasonable and well-motivated applications, though the project does not implement any of them — they motivate the classifier rather than constituting its evaluation.

### 2.4 Theoretical Expectations

For binary text classification on sparse bag-of-words/TF-IDF features, classical theoretical expectations are:

- **Linear models** (Naive Bayes, linear SVM, logistic regression) typically perform comparably and better than nonlinear models, because the sentiment/ideology signal is approximately linear in word presence/absence.
- **Naive Bayes** specifically may be surprisingly strong despite its restrictive independence assumption, especially on short documents where co-occurrence statistics are noisy anyway.
- **Tree-based models** (Random Forest, XGBoost) generally underperform because axis-aligned splits on individual word counts capture only weak per-feature signal.

The actual results closely follow these expectations.

---

## 3. Dataset

### 3.1 Source

The dataset is **"Democrat Vs. Republican Tweets"**, published on Kaggle by Kyle Pastor in 2018. The team cites this source explicitly in both the proposal and the final report.

### 3.2 Composition

| Party | Number of Twitter handles | Number of tweets |
|---|---|---|
| Republican | 222 | 42,434 |
| Democrat | 211 | 42,068 |
| **Total** | **433** | **84,502** |

Approximately 200 latest tweets per user were collected (as of May 2018). Class balance is essentially perfect (50.2% / 49.8%), eliminating the need for class-weight adjustment, oversampling, or threshold tuning.

### 3.3 Schema

Each row contains three columns:

- **`Party`** — categorical label, `'Democrat'` or `'Republican'`.
- **`Handle`** — Twitter screen name of the politician.
- **`Tweet`** — raw tweet text, including `RT` prefixes for retweets, `@username` mentions, hashtags, URLs, and standard punctuation.

Sample raw tweet:

> *RT @NBCLatino: .@RepDarrenSoto noted that Hurricane Maria has left approximately $90 billion in damages.*

This single example illustrates the formatting characteristics that downstream preprocessing must handle: a retweet marker, two `@`-mentions, a punctuation pattern that includes period-mention syntax, and a numeric quantity embedded in the text.

### 3.4 Train/test split

A **stratified 80/20 split** is performed on the cleaned data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    raw_data.cleaned, raw_data.Party,
    stratify=raw_data.Party,
    test_size=0.2,
    random_state=123
)
```

This produces a training set of ~67,602 tweets and a test set of **17,292 tweets** — a sample large enough that test accuracy estimates have a standard error of approximately ±0.003 at 76% accuracy, allowing model accuracies that differ by more than ~0.5 percentage points to be distinguished with confidence.

### 3.5 An Important Caveat — Tweets Per User, Not Per Person

The dataset is collected at the *user* level (200 tweets each from 433 politicians) rather than the *person* level. This means many tweets in the training set come from the same author as some tweets in the test set. Author style is a powerful signal that may leak across the split, potentially inflating apparent accuracy beyond what would generalize to *new* politicians' tweets.

The team does not flag this concern in the report, but it is worth noting as a methodological consideration: a stricter evaluation would split by *author* rather than by *tweet*, ensuring that no author appears in both training and test sets. Real-world deployment performance on tweets from previously-unseen accounts is likely lower than the ~76% reported here.

---

## 4. Methodology — Text Preprocessing Pipeline

The preprocessing pipeline is a four-step transformation from raw tweet text to a clean, stemmed token sequence. Implementation is in `clean.ipynb` (the final code) and a development version exists in `project.ipynb`.

### 4.1 Step 1 — Remove Unrelated Characters

Three classes of token are removed before tokenization, by chained regex substitutions:

```python
def remove_punctuations(t):
    t = re.sub('@\w+', '', t)        # remove @username mentions
    t = "".join([char for char in t if char not in string.punctuation])  # punctuation
    t = re.sub('[0-9]+', '', t)      # digits
    return t
```

Order matters here: the `@\w+` regex must run *before* punctuation stripping, because if `@` were stripped first, `@RepDarrenSoto` would become `RepDarrenSoto` — an indistinguishable proper noun that would then be stemmed and treated as a content word. The team's pipeline correctly handles this dependency.

The example tweet in the report,

> *RT @NBCLatino: .@RepDarrenSoto noted that Hurricane Maria has left approximately $90 billion in damages.*

becomes after Step 1:

> *RT  noted that Hurricane Maria has left approximately  billion in damages*

Note: the `RT` prefix survives Step 1 because no rule explicitly targets it (it lacks an `@` and contains no punctuation/digits). It is removed in Step 2.

### 4.2 Step 2 — Tokenization and Stopword Removal

Tokenization splits on non-word characters using `re.split('\W+', t)`, producing a list of word-only tokens. Stopword removal then filters out:

1. **Standard English stopwords** from NLTK's English stopword list (`the`, `to`, `of`, `and`, `in`, ...).
2. **The Twitter-specific stopword `'rt'`** — explicitly added by the team after observing that "RT" was the 8th most frequent token in both classes.

```python
stopword = nltk.corpus.stopwords.words('english')

def remove_stop_words(t):
    t = [word for word in t if word not in stopword]
    t = [word for word in t if word not in ['rt']]
    return t
```

The team's frequency analysis confirms that the standard stopword list captures the most uninformative tokens. Their reported top-10 frequencies for each class:

| Republican rank | Token | Freq | | Democrat rank | Token | Freq |
|---|---|---|---|---|---|---|
| 1 | the | 37,324 | | 1 | the | 33,980 |
| 2 | to | 28,725 | | 2 | to | 28,825 |
| 3 | of | 15,877 | | 3 | of | 15,483 |
| 4 | and | 15,340 | | 4 | and | 14,704 |
| 5 | in | 13,931 | | 5 | a | 13,076 |
| 6 | a | 12,087 | | 6 | in | 12,074 |
| 7 | for | 11,741 | | 7 | for | 11,382 |
| 8 | rt | 9,986 | | 8 | rt | 9,068 |
| 9 | on | 9,093 | | 9 | is | 7,651 |
| 10 | is | 7,476 | | 10 | on | 7,642 |

**The two distributions are virtually identical** at the top of the frequency tail — confirming that without stopword removal, the most frequent terms carry essentially zero discriminative signal. After stopword removal, more meaningful content words (e.g., *great*, *today*) rise to prominence, as visualized in the team's word-cloud figures.

### 4.3 Step 3 — Stemming with Porter

Stemming reduces morphological variants to a common stem:

```python
ps = nltk.PorterStemmer()
def stemming(t):
    t = [ps.stem(word) for word in t]
    return t
```

Porter stemming is rule-based and aggressive: *senate* → *senat*, *democrats* → *democrat*, *running* → *run*. The output is sometimes a non-word stem (*senat*) but the goal is vocabulary collapse, not interpretability. For BoW/TF-IDF representations consumed by linear models, this is acceptable.

### 4.4 Step 4 — Pipeline Composition and Application

The four steps are composed in a single function and applied row-wise to the dataframe:

```python
def cleaner(t):
    t = remove_punctuations(t).lower()
    t = tokenization(t)
    t = remove_stop_words(t)
    t = stemming(t)
    return t

raw_data['cleaned'] = raw_data.Tweet.apply(lambda x: " ".join(cleaner(x)))
```

The lowercase normalization happens after handle removal but before tokenization, ensuring stopword matching (which uses lowercase reference vocabulary) works correctly. Tokens are joined back into a space-separated string, which is the format expected by `TfidfVectorizer`.

### 4.5 Cleaning example — full pipeline trace

The team provides a step-by-step trace for one tweet, which I reproduce here:

| Step | Output |
|---|---|
| **Raw data** | `Today, Senate Dems vote to #SaveTheInternet.` |
| **Remove unrelated characters** | `Today Senate Dems vote to SaveTheInternet` |
| **Tokenization** | `[Today, Senate, Dems, vote, to, SaveTheInternet]` |
| **Remove stop words** | `[Today, Senate, Dems, vote, SaveTheInternet]` |
| **Stemming** | `[today, senat, dem, vote, savetheinternet]` |

Notable observations:

- The hashtag `#SaveTheInternet` survives as `SaveTheInternet` because the punctuation strip removed only the `#` — useful, since the hashtag carries genuine semantic content.
- The token `to` is correctly identified as a stopword and removed.
- Stemming reduces *Senate* → *senat* and *Dems* → *dem*, collapsing the words into more general forms.

---

## 5. Vectorization — TF-IDF on Tweet-Aware Tokens

After cleaning, tweets are vectorized using **TF-IDF** (Term Frequency – Inverse Document Frequency), with one important customization: the tokenizer is NLTK's `TweetTokenizer`, not the default whitespace tokenizer that `TfidfVectorizer` uses.

### 5.1 TF-IDF — Theoretical framing

TF-IDF assigns each token in each document a weight equal to:

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{1 + \text{df}(t)}
$$

where $\text{tf}(t, d)$ is the count (or normalized frequency) of token $t$ in document $d$, $N$ is the total number of documents, and $\text{df}(t)$ is the number of documents containing $t$.

The intuition is: words that appear in *many* documents (e.g., *today*, *great*) have low IDF and thus low weight regardless of how often they appear in any single tweet. Words that appear in *few* documents (e.g., *gerrymander*, *DREAMer*) have high IDF and thus carry strong discriminative weight when present.

For binary classification on social media text, TF-IDF is the standard representation; raw counts (CountVectorizer) work nearly as well for linear models but are suboptimal for tree-based models, while TF-IDF is more universally appropriate.

### 5.2 The TweetTokenizer choice

```python
from nltk.tokenize.casual import TweetTokenizer
tokenizer = TweetTokenizer(reduce_len=True)

# in the pipeline:
('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
```

`TweetTokenizer` is designed specifically for casual, social-media-formatted English. It correctly handles:

- **Hashtags** — `#savetheinternet` is preserved as a single token rather than being split at the `#`.
- **@-mentions** — although the team has already stripped these in preprocessing, the tokenizer would handle them gracefully if any survived.
- **Emoticons** — `:)`, `:-D` etc. are tokenized as single units.
- **Repeated characters** — the `reduce_len=True` flag normalizes elongated tokens like `sooooo` to `sooo` (truncates 3+ repeats to 3), preventing the model from treating `sooo`, `soooo`, `sooooo` as distinct tokens.

This is a small but methodologically important choice. A naive whitespace tokenizer would produce noisier features and reduce the effective vocabulary's discriminative power.

### 5.3 Pipeline composition

The full vectorization-plus-classification pipeline is encapsulated as a scikit-learn `Pipeline`:

```python
from sklearn.pipeline import Pipeline

nb_pipeline = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
    ('classifier', MultinomialNB())
])
```

The benefit of using `Pipeline` is that **all preprocessing parameters (TF-IDF settings) and classifier hyperparameters can be jointly tuned** by grid search via the `classifier__alpha` parameter naming convention. The pipeline also ensures that training-set TF-IDF parameters (vocabulary, document frequencies) are not contaminated by test-set data — a subtle but important guarantee.

---

## 6. Model 1 — Multinomial Naive Bayes

**Role:** Baseline → ultimately the winning model.

### 6.1 Theoretical framing

Multinomial Naive Bayes is a **generative classifier** that models the conditional distribution of token counts as a multinomial:

$$
P(\mathbf{x} \mid C_k) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_i p_{ki}^{x_i}
$$

where $\mathbf{x}$ is the token-count vector, $C_k$ is the class, and $p_{ki}$ is the probability of token $i$ given class $k$. Taking logs reveals that Multinomial NB is a **linear classifier in log-token-count space**:

$$
\log P(C_k \mid \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^n x_i \log p_{ki} = b + \mathbf{w}^\top \mathbf{x}
$$

where $b = \log P(C_k)$ is the class log-prior and $w_{ki} = \log p_{ki}$. For binary classification, the decision boundary $\log P(C_1 \mid \mathbf{x}) - \log P(C_0 \mid \mathbf{x}) = 0$ is therefore a hyperplane — directly comparable to logistic regression and linear SVM, but with parameters fit by a *closed-form generative procedure* (counting + smoothing) rather than by *discriminative loss minimization*.

This means Naive Bayes:
- **Trains in O(n) time** — single pass over the data to count co-occurrences.
- **Predicts in O(d) time** — a sparse dot product per query.
- **Has only one hyperparameter to tune**: the Laplace smoothing parameter $\alpha$ (added to all counts to prevent zero probabilities for unseen tokens).

### 6.2 Hyperparameter tuning — grid search on $\alpha$

The Laplace smoothing parameter $\alpha$ controls how strongly the model regularizes toward uniform probabilities. Small $\alpha$ trusts observed counts more (lower bias, higher variance); large $\alpha$ smooths more aggressively.

```python
nb_params = [{'classifier__alpha': [3, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]}]
gs = GridSearchCV(estimator=nb_pipeline, param_grid=nb_params,
                  refit=True, cv=4, n_jobs=-1, verbose=10)
gs.fit(X_train, y_train)
```

The grid is logarithmic in scale, spanning eight orders of magnitude. The CV score curve peaks at **α = 0.1** with **CV accuracy 75.94%**, then declines slightly at smaller $\alpha$ (where overconfident smoothing of zero-count tokens hurts) and substantially at larger $\alpha$ (where over-regularization erodes informative tokens).

Selected: **α = 0.1**.

### 6.3 Test performance

The retrained model with α = 0.1 evaluated on the held-out 17,292 tweets:

```
Test Accuracy: 0.757
```

This is the **highest test accuracy of any model in the project**.

### 6.4 Confusion matrix and metrics

| | Predicted Democrat | Predicted Republican |
|---|---|---|
| **Actual Democrat** | 6,012 (TN) | 1,802 (FP) |
| **Actual Republican** | 2,402 (FN) | 7,076 (TP) |

(Treating Republican as the positive class.)

Derived metrics:

| Metric | Naive Bayes |
|---|---|
| Accuracy | 0.757 |
| Precision (Republican) | 0.747 |
| Recall (Republican) | 0.797 |
| F1 score | **0.771** |
| Matthews Correlation Coefficient | **0.514** |

The MCC score of 0.514 (well above zero, where 0 means random) indicates a substantively meaningful relationship between predictions and ground truth, beyond accuracy alone.

### 6.5 Why is Naive Bayes so strong here?

The Multinomial NB independence assumption — that tokens are conditionally independent given the class — is famously incorrect for natural language. Yet on text classification tasks, NB often performs as well as or better than more sophisticated methods. The reasons are well-understood:

1. **The feature space is high-dimensional and sparse.** Most tweets share few tokens, so many co-occurrence statistics that NB ignores are themselves noisy and uninformative.
2. **Per-class token probabilities $p_{ki}$ are well-estimated** with tens of thousands of training examples, even with the independence assumption.
3. **The decision boundary needs to be linear** in log-feature space — and NB fits exactly that, with simple closed-form parameter estimates that avoid optimization difficulties.
4. **Class priors are well-balanced** here, so the prior term $\log P(C_k)$ is uninformative; the discriminative work is done entirely by the per-token log-probabilities.

---

## 7. Model 2 — Linear Support Vector Machine

**Role:** Discriminative max-margin baseline.

### 7.1 Theoretical framing

A linear SVM finds the hyperplane that **maximizes the margin** between the two classes — that is, the hyperplane equidistant from the closest training examples of each class. For a linearly separable problem, this yields a unique optimal classifier; for non-separable problems, a soft-margin formulation introduces a regularization parameter $C$:

$$
\min_{\mathbf{w}, b} \;\; \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^n \max\bigl(0,\; 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\bigr)
$$

The hinge loss penalizes only misclassified points and points within the margin, making the SVM robust to outliers far from the boundary.

For text classification specifically, the team cites Kowalczyk et al. (2020) recommending **linear kernel** for high-dimensional sparse data — kernel methods (RBF, polynomial) generally do not help, because the curse of dimensionality dilutes the signal that nonlinear similarity functions could capture.

### 7.2 Hyperparameter tuning — grid search on C

For a linear-kernel SVM, the only meaningful hyperparameter is the regularization strength $C$. Smaller $C$ = stronger regularization (smoother decision boundary); larger $C$ = less regularization (more closely fits training data).

```python
svm_pipe = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
    ('classifier', SVC(kernel='linear', C=1))
])
svm_params = [{'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]
gs_svm_linear = GridSearchCV(estimator=svm_pipe, param_grid=svm_params,
                             refit=True, cv=4, n_jobs=-1)
```

CV accuracies across the C grid:

| C | CV accuracy |
|---|---|
| 0.0001 | 0.5134 |
| 0.001 | 0.5134 |
| 0.01 | 0.5636 |
| 0.1 | 0.7234 |
| **1** | **0.7555** |
| 10 | 0.7466 |

Two notes:

1. **The transition from random (51%) to useful (76%) happens between C = 0.001 and C = 0.1.** At very small C, the regularization is so strong that the model collapses to predicting a single class.
2. **Performance peaks at C = 1**, with mild degradation at C = 10. The team selects C = 1.

### 7.3 Test performance

```
Test Accuracy: 0.753
```

Approximately 0.4 percentage points below Naive Bayes — a small but consistent gap.

### 7.4 Confusion matrix and metrics

| | Predicted Democrat | Predicted Republican |
|---|---|---|
| **Actual Democrat** | 6,064 (TN) | 1,916 (FP) |
| **Actual Republican** | 2,350 (FN) | 6,962 (TP) |

| Metric | SVM |
|---|---|
| Accuracy | 0.753 |
| Precision (Republican) | 0.748 |
| Recall (Republican) | 0.784 |
| F1 score | 0.765 |
| Matthews Correlation Coefficient | 0.506 |

The SVM is slightly more conservative on the Republican class (recall 0.784 vs. NB's 0.797) but slightly more confident when it does predict Republican (precision 0.748 vs. NB's 0.747 — essentially tied).

### 7.5 Why does SVM slightly underperform NB?

Several plausible reasons:

1. **Hinge loss is more sensitive to outliers than NB's generative procedure.** Tweets that are stylistically idiosyncratic (highly ambiguous tweets, tweets misclassified due to author-specific language) push the hinge loss in ways that the NB log-likelihood ignores.
2. **Naive Bayes' generative bias acts as a regularizer.** With ~67k training examples and tens of thousands of features, the SVM has enough freedom to slightly overfit; NB's class-conditional independence assumption forbids certain types of overfitting that the SVM is structurally able to commit.
3. **Sparse short documents favor count-based models.** Tweets are short (~20–30 tokens after cleaning); for very short documents, the multinomial likelihood is a particularly natural fit.

The 0.4-percentage-point gap is small but, as Section 10 demonstrates, statistically real.

---

## 8. Model 3 — XGBoost

**Role:** Nonlinear ensemble baseline; expected to be competitive on structured data, tested here as a check on whether it transfers to high-dimensional sparse text.

### 8.1 Theoretical framing

XGBoost is a **gradient-boosted decision tree** ensemble. It fits trees sequentially, each new tree minimizing the residual error of the current ensemble:

$$
\text{Obj} = \sum_{i=1}^n \ell\bigl(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr) + \Omega(f_t)
$$

where $\ell$ is the per-example loss and $\Omega(f_t)$ is a regularization term penalizing tree complexity. The team's report includes the closed-form expression for the optimal leaf weight under XGBoost's regularized objective:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

where $G_j$ and $H_j$ are sums of first and second derivatives over the examples in leaf $j$ — this is the additive correction XGBoost applies at each iteration, with $\lambda$ as the L2 regularization strength.

For text classification, gradient-boosted trees are known to underperform linear models because:
- Each tree split is a univariate threshold on a single TF-IDF value, capturing only weak discriminative signal per feature.
- The boosting procedure cannot easily discover interactions across many sparse features simultaneously.
- The number of features (~30k+ vocabulary terms) far exceeds what trees comfortably handle compared to dense numeric data.

### 8.2 Hyperparameter tuning — randomized search

XGBoost has many hyperparameters; rather than running a full grid search (computationally expensive), the team uses `RandomizedSearchCV` to sample configurations from continuous distributions:

```python
import scipy.stats

d = {
    'classifier__max_depth':       [5, 6, 7, 8, 9, 10],
    'classifier__min_child_weight': scipy.stats.uniform(loc=0, scale=10),
    'classifier__eta':              scipy.stats.uniform(loc=0, scale=2),
}

xgb_pipe = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
    ('classifier', XGBClassifier(max_depth=10, n_estimators=300,
                                 learning_rate=0.2, nthreads=-1))
])

rs = RandomizedSearchCV(estimator=xgb_pipe, param_distributions=d,
                        n_iter=10, cv=4, refit=True,
                        random_state=123, n_jobs=-1)
```

The team reports random-search results (showing 10 configurations sampled):

| max_depth | min_child_weight | eta | accuracy |
|---|---|---|---|
| 7 | 2.27 | 1.39 | 0.7035 |
| 8 | 4.91 | 1.10 | 0.7070 |
| 6 | 6.80 | 1.50 | 0.6950 |
| 6 | 1.20 | 4.00 | 0.6980 |
| 6 | 4.30 | 1.06 | 0.6980 |
| 5 | 5.90 | 1.20 | 0.6910 |
| **8** | **1.20** | **1.70** | **0.7100** |
| 9 | 4.20 | 1.03 | 0.6940 |
| 6 | 7.40 | 0.59 | 0.6950 |
| 6 | 6.50 | 0.71 | 0.6940 |

Two patterns the team identifies:

1. **Higher `max_depth` tends to yield higher accuracy.** The best-of-sample at depth 8 outperforms shallower trees.
2. **`min_child_weight` near 1 helps.** This parameter sets the minimum sum of instance weights per leaf; smaller values allow finer-grained leaf partitions.

### 8.3 Final XGBoost configuration

The team selects (slightly reaching beyond the explicit search results):

```python
XGBClassifier(
    max_depth=10, n_estimators=200, learning_rate=1.06,
    min_child_weight=4.27, nthreads=-1
)
```

A few comments on this configuration:

- **`learning_rate=1.06` is unusually high** for XGBoost — typical values are 0.01 to 0.3. Large `eta` causes each tree to make large updates, which combined with `n_estimators=200` produces an aggressive model that risks instability. This choice was emergent from the random search and likely reflects the team's choice to allow eta in [0, 2].
- **`max_depth=10`** is at the high end of the tested range; the random search suggested depths 7–9.
- **`min_child_weight=4.27`** is the value from the configuration that achieved 0.6980 in the random search — a non-optimal sample that the team chose anyway, perhaps because they tuned by hand around the random-search results.

### 8.4 Test performance

```
Test Accuracy: 0.701
```

A clear underperformance — approximately **5 percentage points below Naive Bayes and SVM**. This gap is large enough that no statistical test is needed to confirm XGBoost is meaningfully worse on this task.

### 8.5 Why does XGBoost underperform?

The result is consistent with the well-established pattern that boosted trees do not transfer well to high-dimensional sparse text:

1. **TF-IDF features have weak per-feature signal.** A single token's TF-IDF value is rarely highly discriminative; the discriminative signal lives in the *sum* of weighted contributions across many tokens. Trees split on one feature at a time, fragmenting the data and losing this aggregation.
2. **Sparse representations.** Most TF-IDF entries are zero for any given tweet. Tree splits on "is this token present?" produce highly skewed splits where one branch contains nearly all examples.
3. **Boosting amplifies noise.** With learning rate > 1.0 and 200 trees, the ensemble can amplify noise in the limited training signal, leading to overfitting.
4. **Hyperparameter sensitivity.** With so many hyperparameters (max_depth, eta, min_child_weight, n_estimators, gamma, subsample, colsample), even random search of 10 configurations explores only a tiny corner of the joint space.

For text classification specifically, **the 5-point gap is qualitatively expected**, and the XGBoost result here is in line with broader literature.

---

## 9. Model Comparison and Selection

### 9.1 4-Fold Cross-Validation Comparison

After hyperparameter tuning, the team performs an apples-to-apples comparison using 4-fold cross-validation on the *training* set:

| Model | 4-fold CV Accuracy |
|---|---|
| Naive Bayes | 0.7594 |
| SVM | 0.7643 |
| XGBoost | 0.7007 |

(Stored in `nb_cv_avg`, `svm_cv_avg`, and `xgb_cv_avg_1` respectively.)

Interestingly, **on cross-validation the SVM marginally edges out Naive Bayes** (0.7643 vs. 0.7594). This is a small reversal of the test-set ordering.

### 9.2 Test set comparison

On the held-out 17,292-tweet test set:

| Model | Test Accuracy |
|---|---|
| **Naive Bayes** | **0.7569** |
| SVM | 0.7533 |
| XGBoost | 0.7007 |

On test, Naive Bayes wins by a small margin.

### 9.3 The CV-vs-Test reversal

The CV→test reversal between SVM and Naive Bayes (SVM wins by 0.0049 on CV, loses by 0.0036 on test) is likely a **noise-floor phenomenon**: with stratified test sets of ~17k examples, the standard error on accuracy is ~0.0033, meaning differences smaller than ~0.005 can flip stochastically across train/test partitions. This is precisely what motivates the McNemar's test in Section 10 — the question becomes not "which is bigger?" but "is the difference real?"

### 9.4 The XGBoost gap is decisive

Whatever the SVM-vs-NB ordering, **XGBoost is solidly worse than both** by ~5 percentage points on every metric. The team's decision to deprioritize XGBoost is well-supported by both CV and test results.

### 9.5 Naive Bayes selected as final model

The team's reported final selection rationale:

> *Naive Bayes has the highest accuracy of 76% on our test data set, and the high recall rate of Naive Bayes model allow us to do some potential future works based on such a model. Besides, Naive Bayes model also has lower training and predicting cost, which is potentially useful in industrial applications.*

Three justifications:
1. **Marginally highest accuracy.**
2. **Higher recall (0.797 vs. 0.784).** Particularly important if downstream applications target Republican content specifically — for example, identifying Republican-leaning bots.
3. **Computational efficiency.** NB trains and predicts in milliseconds; SVM training scales poorly with sparse high-dimensional data.

The third point is particularly relevant for industrial deployment, where NB's `O(n + d)` training and `O(d)` prediction make it suitable for streaming or real-time classification of incoming tweets.

---

## 10. McNemar's Test — Are the Models Statistically Different?

The team's most interesting methodological inclusion is a **formal statistical comparison** between Naive Bayes and SVM using McNemar's test. With a 0.4-percentage-point accuracy gap on a 17,292-example test set, it is genuinely unclear whether the difference is meaningful — and "running a statistical test" is the right answer.

### 10.1 The contingency table

McNemar's test requires a 2×2 contingency table of *paired* predictions on the same examples:

```python
from sklearn.metrics.cluster import contingency_matrix

svm_cm = contingency_matrix(svm_pred, nb_pred)
```

Result:

| | NB Predicts Democrat | NB Predicts Republican |
|---|---|---|
| **SVM Predicts Democrat** | 6,758 | 1,222 |
| **SVM Predicts Republican** | 1,056 | 8,256 |

Interpretation:
- **6,758 + 8,256 = 15,014** examples where SVM and NB agree.
- **1,222 + 1,056 = 2,278** examples where they disagree.

Of the disagreements:
- **1,222 cases** where SVM says Democrat and NB says Republican.
- **1,056 cases** where SVM says Republican and NB says Democrat.

### 10.2 The McNemar statistic

McNemar's test asks whether the off-diagonal disagreement counts (1,222 vs. 1,056) are consistent with a null hypothesis of equal error rates. Under the null, the two off-diagonal cells should have approximately equal counts (the disagreements are symmetric); a substantial imbalance indicates one model is systematically right where the other is wrong, more often than the reverse.

Using statsmodels' `mcnemar` with `exact=True`:

```python
from statsmodels.stats.contingency_tables import mcnemar
result_mcnemar = mcnemar(svm_cm, exact=True)
print('statistic=%.3f, p-value=%.3f' % (result_mcnemar.statistic, result_mcnemar.pvalue))
# statistic=1056.000, p-value=0.001
```

Result: **statistic = 1056, p-value = 0.001**.

The p-value of 0.001 is well below the conventional α = 0.05 threshold. **The null hypothesis is rejected.**

### 10.3 Interpretation

Despite the small absolute accuracy gap (0.4 percentage points), Naive Bayes and SVM are making **systematically different predictions** on this dataset. The 1,222 vs. 1,056 imbalance, scaled to a sample of 2,278 disagreements, is statistically significant.

Important caveat: McNemar's test answers *are the models making different errors?*, not *which model is better?*. A significant result here means the two models have different decision boundaries — but to claim NB is *better*, the team relies on the accuracy and F1 differences, which are small but consistently favoring NB.

### 10.4 Why this inclusion matters

Most undergraduate ML projects compare models by raw accuracy and stop there. Including a McNemar's test elevates the comparison from "look which number is bigger" to "is the difference statistically meaningful, and how confident should we be in the choice?" For the specific case of close model performance, this is essential — without the test, the project's ranking could plausibly be a fluke of the train/test split.

---

## 11. Comparative Analysis and Discussion

### 11.1 Headline finding — The simplest model wins

The clearest result of the project is that **Multinomial Naive Bayes, a 1960s-vintage probabilistic model with strong independence assumptions, slightly outperforms a tuned linear SVM and substantially outperforms a tuned XGBoost**. This is consistent with broader text-classification literature but is worth emphasizing: model complexity does not automatically translate to performance, and on text problems specifically, "more sophisticated" often means "more parameters that don't generalize."

### 11.2 The Naive-Bayes / SVM near-tie

The two top models perform within 0.4 percentage points on test accuracy and within 0.5 percentage points on CV. McNemar's test confirms their predictions diverge in a non-random way, so the choice between them is genuine — but small.

For **maximum accuracy**, choose Naive Bayes (just barely).
For **maximum recall on the Republican class**, choose Naive Bayes (0.797 vs. 0.784).
For **fastest inference**, choose Naive Bayes (closed-form prediction).
For **interpretability of feature contributions**, both linear models are equally interpretable in terms of per-feature weights.

There is no obvious advantage to choosing SVM over Naive Bayes here, except possibly in pipeline integration with kernel-based methods that the project does not pursue.

### 11.3 The XGBoost underperformance

XGBoost's 5-percentage-point gap behind the linear models is the most pedagogically useful negative result in the project. It is consistent with the canonical advice that **gradient-boosted trees underperform on high-dimensional sparse text representations**, and provides a clean empirical illustration of why bag-of-words text problems remain a domain where classical methods are not just baselines but competitive solutions.

If the team wanted to make XGBoost competitive, they would need to either:
1. **Reduce dimensionality** via PCA, LSA, or feature selection.
2. **Replace BoW/TF-IDF with dense embeddings** (Word2Vec, GloVe, sentence embeddings) — a much smaller dense input where tree splits make more sense.
3. **Use a fundamentally different boosted model** (e.g., LightGBM with sparse-aware splits, or an embedding-based deep model).

None of these are pursued here, which is reasonable given the project's scope.

### 11.4 The methodological strengths

Several aspects of the project deserve specific praise:

1. **Tweet-aware tokenization with `TweetTokenizer`.** A small but careful detail that the team got right.
2. **Order-aware preprocessing.** Removing `@username` mentions before punctuation stripping is the correct sequence; reversing the order would corrupt the data.
3. **Pipeline-based hyperparameter search.** Using `Pipeline` ensures that TF-IDF parameters are correctly scoped to training folds and not contaminated by validation/test data.
4. **Multi-metric evaluation.** Reporting accuracy, precision, recall, F1, and MCC is more than most undergraduate projects do, and gives a fuller picture of model behavior.
5. **McNemar's test inclusion.** A genuinely useful methodological contribution that elevates the model comparison.

### 11.5 The methodological weaknesses

A few weaknesses worth flagging:

1. **Tweet-level rather than author-level split.** As discussed in Section 3.5, the same author appears in both training and test sets. Test accuracy of 76% likely overstates performance on new authors' tweets — possibly substantially. The team does not flag this concern.

2. **No baseline beyond the three models tested.** A logistic regression baseline would have been informative — it would likely match or slightly underperform Naive Bayes and is the most-often-cited classical baseline for text classification. Its omission is a small gap.

3. **XGBoost hyperparameter search was incomplete.** With only 10 random configurations and an unusual `learning_rate=1.06`, the XGBoost result may be more pessimistic than necessary. A more careful search with `eta` constrained to typical ranges (0.01–0.3) might have closed some of the gap to the linear models — though probably not all of it.

4. **No error analysis.** The team reports confusion matrices but does not investigate *which tweets* the models get wrong, or whether the errors fall into systematic categories (e.g., bipartisan tweets, retweets of opposite-party content, ambiguous topical statements). This would have been a valuable qualitative complement to the quantitative metrics.

5. **No examination of the most predictive features.** For Naive Bayes, $\log P(\text{token} | \text{Republican}) - \log P(\text{token} | \text{Democrat})$ is a natural interpretability measure that would show which words most strongly indicate each class. This would not only validate the model but also produce a substantively interesting linguistic finding.

6. **The comment "We will test all three of our models to see the result of 0.632, 0.632+, cv and test result"** appears in the notebook's discussion section but is never followed up on. The .632+ bootstrap estimator is a standard alternative to cross-validation; including it would have addressed CV vs. .632+ comparison cleanly.

### 11.6 What this project does and doesn't tell us

**It tells us:**
- Multinomial Naive Bayes is a strong choice for tweet-level political-ideology classification.
- The performance gap to a tuned SVM is small but statistically real.
- XGBoost is decisively worse on this representation.

**It doesn't tell us:**
- How well the model generalizes to *unseen authors* — the train/test split shares authors.
- How well the model generalizes to *non-political-figure tweets* — the data is from elected representatives, not the general public.
- How the model would compare to **modern neural baselines** — fine-tuned BERT/RoBERTa would likely achieve 80–85% on this task, though at much higher computational cost.
- Which **specific linguistic features** drive the classification — a key piece of the interpretation puzzle.

---

## 12. Conclusions and Future Work

### 12.1 Conclusions

This project's principal conclusion is a clean, concrete one: **Multinomial Naive Bayes with TF-IDF features and Porter stemming achieves 76% test accuracy on tweet-level Democrat/Republican classification, slightly outperforming a tuned linear SVM and substantially outperforming a tuned XGBoost.** The result is supported by:

- A consistent preprocessing pipeline that handles Twitter-specific text features correctly (handles, hashtags, retweet markers, casual punctuation).
- Cross-validated hyperparameter selection for each model.
- Multi-metric evaluation including precision, recall, F1, and MCC.
- A formal McNemar's test confirming that the small NB-vs-SVM gap reflects systematically different predictions, not noise.

The project's secondary finding is that **gradient boosting underperforms linear models on this representation**, consistent with broader text-classification literature and supporting the conventional advice that classical methods remain competitive baselines for bag-of-words/TF-IDF problems.

### 12.2 Future Work — Acknowledged in the Report

The team's final report acknowledges potential applications:

1. **Public-opinion monitoring at scale.** Apply the trained classifier to a sample of the broader Twitter user base to track partisan-content distribution over time.
2. **Political bot identification.** Identify accounts whose tweet stream is unusually strongly classified as one ideology — a possible signal of automated partisan content.
3. **Demographic mapping of political ideology.** Combine the classifier with user metadata to produce geographic distributions of partisan content density.
4. **Predicting the 2020 election.** Use partisan-content prevalence as one input to election prediction models.

### 12.3 Future Work — Not in the Report But Implied

Several methodological extensions would meaningfully strengthen the project:

5. **Author-level train/test split.** Repeat the experiments with no author appearing in both training and test sets. This would give a more honest estimate of generalization to new accounts.

6. **Logistic regression baseline.** A standard inclusion for text classification, expected to match or slightly trail Naive Bayes.

7. **Transformer-based baselines.** Fine-tuned BERT, DistilBERT, RoBERTa, or modern instruction-tuned LLMs would likely push accuracy to 80–85% at the cost of substantially higher computational requirements. Including such a baseline would contextualize the gap between classical NLP and modern systems.

8. **N-gram features.** Extending TF-IDF from unigrams to bigrams (and possibly trigrams) typically yields a 1–3 percentage point gain on text classification tasks. This would be a low-cost extension to test.

9. **Error analysis.** Examine the misclassified tweets manually. Are they short tweets without political content? Tweets quoting opposite-party figures? Bipartisan tweets? Disambiguating error categories often suggests further preprocessing or modeling improvements.

10. **Feature interpretation.** Compute per-token log-likelihood-ratios for Naive Bayes; identify the 50 most Republican-indicative tokens and the 50 most Democrat-indicative tokens. This is both a model-validity check (do the words make sense?) and a substantive linguistic finding.

11. **Apply .632+ bootstrap evaluation**, which the notebook flagged as a discussion item but did not implement.

12. **Calibration analysis.** For Naive Bayes specifically, raw predicted probabilities are often poorly calibrated (overconfident). A reliability diagram + Platt scaling or isotonic recalibration could improve probabilistic outputs without changing classification accuracy.

13. **Out-of-distribution evaluation.** The original proposal mentioned testing the model on news articles and political speeches. This was not implemented in the final report, but would be a valuable sanity check on whether the tweet-trained model captures partisan content broadly or merely tweet-stylistic markers of party affiliation.

---

## 13. Appendix — File Inventory and Hyperparameter Tables

### 13.1 File Inventory

| File | Type | Role |
|---|---|---|
| `Proposal_for_Twitter_Posts_Political_Ideology_Classification.pdf` | PDF | Original 2-page project proposal |
| `FinalReport.pdf` | PDF | Final 7-page project report (the principal write-up) |
| `report.pdf` | PDF | LaTeX template for the final report — not a substantive contribution |
| `clean.ipynb` | Notebook | **Final code** (per the README) — full pipeline from raw data to final evaluation |
| `project.ipynb` | Notebook | Exploratory/development notebook predating `clean.ipynb` |
| `README.md` | Markdown | Brief project description with link to FinalReport.pdf |

### 13.2 Final Hyperparameters

| Model | Hyperparameter | Value |
|---|---|---|
| Multinomial Naive Bayes | `alpha` (Laplace smoothing) | 0.1 |
| Linear SVM | `kernel` | linear |
| Linear SVM | `C` | 1 |
| XGBoost | `max_depth` | 10 |
| XGBoost | `n_estimators` | 200 |
| XGBoost | `learning_rate` (eta) | 1.06 |
| XGBoost | `min_child_weight` | 4.27 |

### 13.3 Software Stack

- **Python 3** with NumPy, pandas, matplotlib, seaborn
- **scikit-learn**: `Pipeline`, `TfidfVectorizer`, `MultinomialNB`, `SVC`, `RandomForestClassifier`, `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, `train_test_split`
- **xgboost**: `XGBClassifier`
- **NLTK**: `stopwords`, `PorterStemmer`, `TweetTokenizer`
- **mlxtend**: `plot_confusion_matrix`
- **statsmodels**: `mcnemar` for the McNemar's test
- **scipy.stats**: continuous distributions for randomized hyperparameter search

### 13.4 Hardware

The team reports running training and tuning on personal laptops (MacBook Pro 15" 2018, Dell XPS 13). XGBoost's randomized search was the most computationally expensive step but completed within reasonable time on consumer hardware — a useful demonstration that classical text classification at ~85k examples is well within the reach of a single laptop without specialized hardware.

### 13.5 Final Performance Reference Table

| Metric | Naive Bayes | SVM | XGBoost |
|---|---|---|---|
| 4-fold CV Accuracy | 0.7594 | 0.7643 | 0.7007 |
| Test Accuracy | **0.7569** | 0.7533 | 0.7007 |
| Test Precision (Republican) | 0.747 | 0.748 | — |
| Test Recall (Republican) | **0.797** | 0.784 | — |
| Test F1 (Republican) | **0.771** | 0.765 | — |
| Test MCC | **0.514** | 0.506 | — |

McNemar's test (NB vs. SVM): **statistic = 1056, p-value = 0.001** → reject H₀ of equal error rates.

---

*End of report.*
