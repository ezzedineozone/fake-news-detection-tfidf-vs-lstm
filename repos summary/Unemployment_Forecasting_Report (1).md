# Unemployment Rate Forecasting using Machine Learning

**Project Report — STAT 451, Fall 2020 Final Project (Group 3)**

**Authors of original work:** Susan Jiao, Yuanhang Wang, Yi Xiao
**Advisor:** Dr. Sebastian Raschka
**Data source:** Federal Reserve Economic Data (FRED-MD), Jan 1959 – Aug 2020

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals and Motivation](#2-project-goals-and-motivation)
3. [Dataset and Exploratory Data Analysis](#3-dataset-and-exploratory-data-analysis)
4. [Feature Engineering — Sliding-Window Time-Series Construction](#4-feature-engineering--sliding-window-time-series-construction)
5. [Feature Selection](#5-feature-selection)
6. [Model 1 — Linear Regression](#6-model-1--linear-regression)
7. [Model 2 — Random Forest](#7-model-2--random-forest)
8. [Model 3 — XGBoost](#8-model-3--xgboost)
9. [Model 4 — LSTM](#9-model-4--lstm)
10. [Model 5 — ARIMA](#10-model-5--arima-comparative-baseline-only)
11. [Comparative Analysis and Discussion](#11-comparative-analysis-and-discussion)
12. [Conclusions and Future Work](#12-conclusions-and-future-work)
13. [Appendix — File Inventory and Indicator Descriptions](#13-appendix--file-inventory-and-indicator-descriptions)

---

## 1. Executive Summary

This project investigates **single-step monthly unemployment rate forecasting** using the FRED-MD macroeconomic database, comparing four supervised machine learning approaches — **Linear Regression**, **Random Forest**, **XGBoost**, and **LSTM** — alongside a traditional **SARIMA** statistical baseline. The forecasting target is the U.S. civilian unemployment rate (UNRATE), and the predictive horizon is one month ahead, conditioning on the previous three months of either (a) UNRATE alone, or (b) eleven correlation-selected economic indicators.

The project addresses a methodologically subtle question: **on a relatively simple single-step forecast, does additional model capacity (ensembles, deep learning) actually improve performance over a well-specified linear baseline?** The answer obtained is — counterintuitively but interpretably — **no**.

**Headline results, January 1959 – December 2019 (excluding the COVID-19 distortion):**

| Model | Features | Test MSE | Test MAE |
|---|---|---|---|
| **Linear Regression** | 1 feature (UNRATE only) | **0.028** | **0.130** |
| Random Forest | 1 feature | 0.035 | 0.144 |
| XGBoost | 1 feature | 0.054 | 0.171 |
| Linear Regression | 11 features | 0.094 | 0.245 |
| **XGBoost** | 11 features | **0.055** | **0.190** |
| Random Forest | 11 features | 0.191 | 0.382 |
| LSTM (3-month window) | 11 features | 0.489 | 0.610 |
| LSTM (1-month window) | 11 features | 0.067 | 0.231 |

**Two complementary findings emerge:**

1. **Among single-feature models**, plain Linear Regression wins decisively. Recent unemployment is so autocorrelated that its near-future value is approximately a linear function of the immediate past — additional model capacity captures noise rather than signal.

2. **Among multi-feature models**, XGBoost wins. Once exogenous economic indicators are introduced, the relationship between predictors and target is no longer well-described by an unregularized linear model with 33 features, and gradient boosting's bias–variance tradeoff is appropriate.

A separate experiment using year-2020 data (capturing the COVID-19 unemployment spike) tests whether models trained on 1959–2019 generalize to a regime-shift event. They do not — every model degrades severely — but the degradation is least catastrophic for XGBoost and the LSTM, suggesting these architectures retain some representational capacity even under distribution shift.

---

## 2. Project Goals and Motivation

### 2.1 Policy and Research Context

Unemployment rate is among the most consequential economic indicators a government tracks. It directly informs:

- **Monetary policy** — the Federal Reserve adjusts interest rates partly in response to labor-market slack.
- **Fiscal and welfare policy** — unemployment insurance benefit duration is automatically extended in recessions; calibrating the extension period is a quantitative question whose accuracy depends on forecasts.
- **Recession detection** — change in unemployment is a key component of the Sahm Rule and other real-time recession indicators.

Traditional time-series forecasting techniques (ARMA, ARIMA, VARMA, threshold autoregressive models) are widely used but often produce unsatisfactory results, particularly during regime changes. More recent literature — including Cook & Hall (2017) at the Federal Reserve Bank of Kansas City — explores deep neural network architectures (in particular encoder-decoder LSTMs) that are reported to outperform consensus forecasting at short horizons. This project sits within that research thread.

### 2.2 Specific Objectives

1. **Construct a supervised-learning representation** of a multivariate time-series forecasting problem using a sliding-window transformation, suitable for both classical ML and recurrent neural architectures.
2. **Empirically compare four model families** — Linear Regression, Random Forest, XGBoost, LSTM — on identical train/test splits for direct, controlled comparison.
3. **Test two feature regimes**:
   - **Univariate**: condition only on past values of the target (UNRATE) — a pure autoregressive setup.
   - **Multivariate**: condition on UNRATE plus 10 correlation-selected economic indicators.
4. **Test two data regimes**:
   - **Pre-pandemic only** (Jan 1959 – Dec 2019) — captures normal economic dynamics.
   - **Including 2020 pandemic data** — tests model robustness to an unprecedented regime shift.
5. **Include a SARIMA baseline** as a representative of classical statistical time-series forecasting — but interpret it carefully, since its objective function (likelihood / AIC) differs from the supervised-learning models' objective (one-step-ahead MSE).

### 2.3 Theoretical Expectation

The literature strongly suggests deep learning should win for time-series forecasting, particularly with multivariate inputs. However, two countervailing forces are operative here:

- **Limited sample size.** Monthly data from 1959 to 2019 yields ~730 examples — small by deep-learning standards, where datasets are typically in the millions.
- **Simple task structure.** Single-step forecasting on a slowly-evolving process (unemployment changes by ~0.1% per month in normal times) is not a task that demands flexibility — it demands a small number of well-tuned weights.

This tension is precisely what the experimental design probes.

---

## 3. Dataset and Exploratory Data Analysis

### 3.1 Source

The data come from **FRED-MD** (Federal Reserve Economic Data — Monthly Database), curated by McCracken at the Federal Reserve Bank of St. Louis. The file `Econ_predictors_2020-08.csv` contains **129 economic indicators** sampled monthly from January 1959 through August 2020 (740 rows × 130 columns including the date).

### 3.2 Indicator Grouping

The team partitioned the 129 indicators into eight semantic groups for EDA, following standard FRED-MD conventions:

| Group | Theme | Example indicators |
|---|---|---|
| 1 | Income / Output | RPI, INDPRO, IPFPNSS |
| 2 | Labor market | **UNRATE**, CLAIMSx, HWIURATIO, UEMPMEAN, UEMPLT5 |
| 3 | Housing | HOUST, PERMIT |
| 4 | Consumption | DPCERA3M086SBEA, RETAILx, UMCSENTx |
| 5 | Money & credit | M1SL, M2SL, BUSLOANS |
| 6 | Interest rates / spreads | FEDFUNDS, GS10, BAAFFM |
| 7 | Prices | CPIAUCSL, PPICMM, OILPRICEx |
| 8 | Stock market | S&P 500, S&P PE ratio, VXOCLSx |

### 3.3 Missing-Data Handling

A simple but principled policy was adopted: **drop rows or columns with missing values rather than impute**. The justification is twofold:

1. **Most missing values are consecutive runs**, e.g., an indicator with no data prior to Jan 1960. Such gaps cannot be reasonably interpolated.
2. **Imputation by mean/median is dangerous in this setting** because (a) economic indicators have large fluctuations, and (b) only 3 months of features are used, so any single imputed value carries disproportionate weight in the prediction.

Concretely:
- The last row (Aug 2020) had widespread missingness due to release-lag and was dropped.
- For the 11 selected features, only a few rows had missing values, and these were dropped without significant loss.

### 3.4 Visual Inspection of UNRATE

```python
plt.plot(predictors.sasdate, predictors.UNRATE)
plt.title("Unemployment rate over time")
```

The time-series plot reveals:

- A **slow trend component** with values ranging roughly 3.5%–11% over six decades.
- **Cyclical recessions** producing sharp spikes (notably 1975, 1982, 1992, 2001, 2009).
- An **extreme outlier in April 2020** (~14.7%) caused by COVID-19 lockdowns — qualitatively different from any other observation in the series.

The 2020 outlier motivates the dual experimental regime (with/without 2020).

### 3.5 Correlation Analysis Per Group

Pearson correlation matrices were computed within each indicator group, joining UNRATE for cross-correlation inspection. Findings:

| Group | Indicators with |corr| > 0.5 vs. UNRATE |
|---|---|
| 1 — Income/Output | CUMFNS (capacity utilization, manufacturing) |
| 2 — Labor market | Many (by construction): CLAIMSx, HWIURATIO, UEMPLT5, UEMP5TO14, UEMP15OV, UEMP15T26, UEMP27OV, UEMPMEAN, CES1021000001 |
| 3 — Housing | None strongly correlated |
| 4 — Consumption | UMCSENTx (consumer sentiment, negatively correlated) |
| 5 — Money | None strongly correlated |
| 6 — Interest | BAAFFM (Baa corporate bond spread over fed funds) |
| 7 — Prices | None strongly correlated |
| 8 — Stocks | None strongly correlated |

The fact that the labor-market group dominates the correlation rankings is unsurprising — many of those indicators are definitionally linked to unemployment (e.g., UEMP5TO14 is the count of people unemployed for 5–14 weeks). Of greater interest is the appearance of **CUMFNS** (manufacturing capacity utilization, a real-economy measure) and **BAAFFM** (a credit-spread measure, often a leading recession indicator) as out-of-group strong correlates.

---

## 4. Feature Engineering — Sliding-Window Time-Series Construction

### 4.1 The Sliding-Window Transformation

Time-series forecasting is reformulated as a **supervised regression problem** by constructing input/output pairs from a sliding window over the time series. Given `m` months of data with `n` indicators, define:

- **Input**: a flattened concatenation of all `n` indicators across `w` consecutive months (length `n × w`).
- **Output**: the value of UNRATE in the month *immediately following* the input window.

For `w = 3` (the default in this project):

$$
\mathbf{x}^{(i)} = \bigl[d_1^{(i)}, \ldots, d_n^{(i)}, \; d_1^{(i+1)}, \ldots, d_n^{(i+1)}, \; d_1^{(i+2)}, \ldots, d_n^{(i+2)}\bigr] \in \mathbb{R}^{3n}
$$

$$
y^{(i)} = d_{\text{UNRATE}}^{(i+3)}
$$

This produces `m − 3` training examples. With `n = 11` features and `m ≈ 730`, this gives ~727 examples with 33 input features each.

### 4.2 Implementation — `Window_generator` class

The team implemented this transformation as a scikit-learn-compatible transformer:

```python
class Window_generator(BaseEstimator, TransformerMixin):
    def __init__(self, input_window_length, label_width=1, shift=1,
                 label_columns=["UNRATE"]):
        self.input_window_length = input_window_length
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns
        self.total_window_length = input_window_length + label_width + shift - 1

    def fit(self, X, y=None):
        self.start_indices = [i for i in range(X.shape[0] - self.total_window_length + 1)]
        self.end_indices = [i + self.input_window_length - 1 for i in self.start_indices]
        self.label_start = [i + self.input_window_length + self.shift - 1 for i in self.start_indices]
        self.label_end = [i + self.total_window_length - 1 for i in self.start_indices]
        ...

    def transform(self, X, y=None):
        input_window, label_window = self.create_window_data()
        input_flatten = input_window.reshape(input_window.shape[0], -1)
        label_flatten = label_window.reshape(label_window.shape[0], -1)
        ...
```

The class is parametrized by:
- `input_window_length` (= 3) — number of months of history used as input
- `label_width` (= 1) — number of months to predict
- `shift` (= 1) — lookahead distance from end of input to start of label
- `label_columns` — which column(s) to treat as the target

A parallel utility, `series_to_supervised`, was also implemented (adapted from a Brownlee blog post) — it produces an equivalent transformation with slightly different bookkeeping and is used in the Random Forest and XGBoost notebooks.

### 4.3 Train / Validation / Test Protocol

A critical methodological constraint applies: **k-fold cross-validation is invalid for time-series forecasting**, because a fold drawn from the future cannot be used to predict a fold drawn from the past without temporal leakage.

Instead, the team uses a **chronological holdout split** with a `PredefinedSplit` to drive scikit-learn's GridSearchCV machinery:

```python
# Train/val/test split: 64/16/20
rf_train_val_df, rf_test_df = train_test_split(data, test_size=0.2, shuffle=False)
rf_train_df, rf_val_df = train_test_split(rf_train_val_df, test_size=0.2, shuffle=False)

# Construct a PredefinedSplit so GridSearchCV uses our chronological val set
rf_split = [-1 for _ in range(rf_train_X.shape[0])] + [0 for _ in range(rf_val_X.shape[0])]
rf_cv = PredefinedSplit(rf_split)
```

The `-1` markers indicate "always in training", and the `0` markers indicate "in this validation fold". This is the correct way to integrate temporal holdout into scikit-learn's hyperparameter search infrastructure.

### 4.4 Standardization

Features and target are both standardized using `StandardScaler` fit on the training set:

```python
scaler = StandardScaler()
scaler.fit(df_train.values)
df_train_scaled = pd.DataFrame(scaler.transform(df_train.values), columns=df_train.columns)
df_test_scaled = pd.DataFrame(scaler.transform(df_test.values), columns=df_test.columns)
```

Predictions are inverted to original scale via `prediction * y_std + y_mean` before computing MSE/MAE. Crucially, the scaler is fit *only on training data* — a small but important detail to prevent test-set leakage through the scaling parameters.

---

## 5. Feature Selection

### 5.1 Correlation-Based Initial Selection

The team's first selection criterion was the **absolute correlation** of each indicator with UNRATE, with a threshold of 0.5. This produced an initial set of **10 economic indicators** plus UNRATE itself:

```
['CUMFNS', 'BAAFFM', 'CES1021000001', 'CLAIMSx', 'HWIURATIO',
 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV', 'UEMP15T26', 'UEMP27OV',
 'UNRATE']
```

(Full descriptions in Appendix.) Notable observations on this selection:

- **Five UEMP* features** are different stratifications of unemployment duration — they are essentially decompositions of UNRATE itself, so their high correlation is partially mechanical.
- **CES1021000001** (mining and logging employment) is an unusual inclusion — its strong correlation likely reflects the cyclicality of resource-extraction employment.
- **BAAFFM** (corporate bond spread) is the most economically interesting selection — credit spreads are a recognized leading indicator of recessions.
- **CUMFNS** (manufacturing capacity utilization) is a core real-economy measure.

### 5.2 Random-Forest Importance — A Confirmatory Check

As a second-pass check, a Random Forest regressor was fitted on the 11-feature input window (33 features after sliding-window expansion) and feature importances inspected:

```python
top10
#                      0
# UNRATE_1m     0.937987
# UNRATE_2m     0.027582
# UEMP15T26_1m  0.004939
# UEMP5TO14_2m  0.004427
# CUMFNS_1m     0.003219
# UEMP5TO14_3m  0.003042
# HWIURATIO_1m  0.002753
# CLAIMSx_1m    0.001652
# UEMP15T26_2m  0.001222
# CLAIMSx_2m    0.001191
```

**More than 95% of total importance accrues to UNRATE from the previous month alone.** This is the single most consequential empirical fact in the project — it tells us that:

1. **Unemployment is highly autocorrelated month-to-month**: knowing the value last month is overwhelmingly predictive of next month's value.
2. **All other features collectively contribute < 5% predictive power** under a tree-based model.
3. Any improvement from the 10 supplementary features must come from the *small* fraction of variance not already captured by autoregression.

This finding motivates the team's two parallel feature regimes (1-feature vs. 11-feature) and explains many of the downstream results.

### 5.3 An Alternative Approach — Seasonal Differencing + Lagged UNRATE

A separate exploratory line in the Random Forest section applies **12-month seasonal differencing** to remove low-frequency trend, then lags the differenced series by 1–12 months:

```python
differenced = series.diff(12)
differenced = differenced[12:]
# Build lag features t-12, t-11, ..., t-1, target = t
for i in range(12, 0, -1):
    dataframe['t-'+str(i)] = series1.UNRATE.shift(i)
dataframe['t'] = series1.UNRATE
```

Random Forest importances on this transformed data confirm that **t-1** alone receives ~94% of the importance, with the remaining 6% spread across lags 2–12. A small comparative R² benchmark on this representation gave: Linear Regression (0.952), Gradient Boosting (0.948), Random Forest (0.941), KNN (0.940), Decision Tree (0.926) — confirming Linear Regression's surprising dominance even on a seasonally-differenced representation.

---


## 6. Model 1 — Linear Regression

**Role:** Baseline, ultimately winning model in single-feature regime.

### 6.1 Theoretical framing

Ordinary Least Squares Linear Regression fits

$$
y = \mathbf{w}^\top \mathbf{x} + b
$$

by minimizing $\|y - \hat{y}\|_2^2$ in closed form. It is the simplest possible regression model and serves multiple roles in this project: a sanity-check baseline, a calibration reference for more complex models, and (as it turns out) the actual best performer in the univariate regime.

For a forecasting problem of the form "predict UNRATE next month given UNRATE in the previous 3 months", the linear model is essentially fitting an **AR(3) autoregression** by least squares — an entirely classical specification.

### 6.2 Experiment 1 — All 129 features (baseline check)

A naive first experiment used *all* 129 economic indicators across 3 months as features (giving 387 predictors). The result is diagnostic:

```
Train MSE: 0.0075
Test MSE: 162.547
```

The catastrophic test error — over 20,000× the training error — is a textbook demonstration of **overfitting in a high-dimensional, low-sample-size regime** (387 predictors, ~580 training points). The training residual is small because the model has near-enough degrees of freedom to interpolate; the test residual explodes because the fitted weights are not generalizable. This experiment serves principally as motivation for feature selection.

### 6.3 Experiment 2 — 11 features, 2020 excluded

Using the 11 correlation-selected features (33 input dimensions after the 3-month window):

```
Train MSE: 0.0210, Train MAE: 0.112
Test MSE:  0.0940, Test MAE: 0.246
```

A clear gap between train and test MSE (factor of ~4.5) indicates moderate overfitting. With 33 features and ~580 training points, OLS still has enough flexibility to memorize idiosyncratic patterns. The dominant test-time errors are coefficient-instability errors: coefficients fitted to one regime do not transfer perfectly to another.

The largest fitted coefficients are diagnostic:

```
PCEPI_2m            45.08
DSERRG3M086SBEA_1m  34.52
UEMP15OV_1m         30.27
UEMP15OV_3m         25.96
INDPRO_2m           15.15
```

Note that PCEPI and DSERRG3M086SBEA appear here despite *not* being in the 11-feature set — this listing is from the 129-feature baseline. In the 11-feature run, the coefficients on the various UEMP* duration features dominate, as expected.

### 6.4 Experiment 3 — 1 feature (UNRATE only), 2020 excluded — **WINNING MODEL**

Restricting to UNRATE alone (3 input dimensions: UNRATE_t-2, t-1, t-0):

```
Train MSE: 0.0295, Train MAE: 0.130
Test MSE:  0.0278, Test MAE: 0.130
```

This is a striking result: **test MSE is *lower* than train MSE**. Two facts converge to make this happen:

1. **The model has effectively 3 weights** — practically impossible to overfit on 580+ training points.
2. **The chronological holdout produces a test set whose autocorrelation structure happens to be slightly more linearly predictable than the training set's.** This is not a guarantee — it's an artifact of the specific period 2016–2019 being unusually smooth.

This 0.0278 is the **best single-step test MSE achieved by any model in the project**.

### 6.5 Experiments 4–5 — Including 2020 data

When 2020 data are included in the test set (2018–2020 instead of 2018–2019), every model degrades:

| Configuration | Test MSE | Test MAE |
|---|---|---|
| LR, 11 features, 2020 included | 0.837 | 0.365 |
| LR, 1 feature, 2020 included | 0.938 | 0.250 |

The factor of ~30× increase in MSE is dominated entirely by the ~9% jump in unemployment in April 2020 — a single month produces most of the residual. The MAE is more informative here because it is less sensitive to the single outlier; under MAE, the 11-feature model degrades from 0.246 to 0.365 (a factor of 1.5), much more reasonable.

### 6.6 Why does 1-feature outperform 11-feature?

This is the most interesting question raised by these results. Two complementary explanations:

1. **Statistical**: 33 features × ~580 examples gives a ratio that's still in the danger zone for OLS. Coefficient variance is meaningful, and noise in the supplementary features bleeds into the prediction. Three features × 580 examples is essentially noiseless.

2. **Structural**: The feature-importance analysis showed that >95% of predictive power lives in UNRATE_t-1. The other 10 features collectively offer at most 5% of variance reduction — and OLS cannot weigh them efficiently against the dominant feature. A regularized linear model (Lasso/Ridge) would likely perform comparably to the 1-feature OLS, automatically zeroing out the supplementary features.

This finding generalizes: **adding correlated, weakly informative features to OLS without regularization tends to *hurt* generalization**, even if those features are correlated with the target in isolation.

---

## 7. Model 2 — Random Forest

**Role:** Capture nonlinear relationships and feature interactions; provide an ensemble baseline against XGBoost.

### 7.1 Theoretical framing

A Random Forest regressor is an ensemble of decision trees trained on bootstrap samples with random feature subsetting at each split. Predictions are the average of the constituent trees' predictions. The two sources of randomness — bagging and per-split feature randomization — decorrelate the trees and reduce variance.

For time-series forecasting, Random Forest's main appeal is its ability to capture **threshold effects** and **feature interactions** that linear models cannot. Its main weakness is that, like all tree models, it cannot extrapolate beyond the range of values seen in training — a serious limitation for trended series like unemployment.

### 7.2 Hyperparameter Tuning — Two-Stage Search

A two-stage search was performed, following the standard "RandomizedSearchCV → GridSearchCV" pattern:

**Stage 1: RandomizedSearchCV** over 100 random combinations of:

```python
random_grid = {
    'n_estimators':      [200, 400, ..., 2000],
    'max_features':      ['auto', 'sqrt'],
    'max_depth':         [10, 20, ..., 110, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'bootstrap':         [True, False]
}
```

**Stage 2: GridSearchCV** over a refined grid centered on the random-search best:

```python
rf_param_grid = {
    'bootstrap':         [True],
    'max_depth':         [100, 110, 120, 130],
    'max_features':      [2, 3],
    'min_samples_leaf':  [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators':      [100, 200, 300, 1000]
}
```

The CV strategy uses `PredefinedSplit` to enforce chronological train→validation ordering, as discussed in Section 4.3.

### 7.3 Experiment 1 — Random Forest with 1 feature (UNRATE only), 2020 excluded

After tuning, the best parameters from RandomizedSearchCV were:

```
{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4,
 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True}
```

Test results:

```
Train MSE: 0.0217, Train MAE: 0.110
Test MSE:  0.0353, Test MAE: 0.144
```

This is **worse than Linear Regression's 0.0278** — by roughly 27%. The Random Forest is paying a complexity cost for capacity it cannot productively use on a near-linear three-feature input.

### 7.4 Experiment 2 — Random Forest with 11 features, 2020 excluded

After two-stage tuning, the best parameters were:

```
{'bootstrap': True, 'max_depth': 100, 'max_features': 5,
 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 1200}
```

Test results:

```
Train MSE: 0.0107, Train MAE: 0.073
Test MSE:  0.1905, Test MAE: 0.383
```

Two observations:

1. **Severe overfitting.** Train MSE (0.011) is nearly 18× lower than test MSE (0.190).
2. **The 11-feature Random Forest underperforms the 1-feature Random Forest** by a factor of ~5.4 on test MSE.

The combination is diagnostic: the additional features supply additional capacity that the model uses to memorize training noise rather than extract signal. With 33 input features, even shallow trees can find spurious axis-aligned patterns that fail to generalize.

### 7.5 Underperformance vs. validation MSE

The team flags an important methodological issue: the Random Forest's *validation* MSE during hyperparameter tuning was very low (~0.005), but test MSE is much higher (~0.190). This **distribution shift between validation and test sets** is a recurring theme throughout the project. The validation set covers ~2014–2017 (a stable, low-unemployment regime); the test set covers ~2018–2019, which is similarly stable but stylistically different. With only one validation fold available (cross-validation being temporally invalid), tuning becomes brittle.

---

## 8. Model 3 — XGBoost

**Role:** State-of-the-art gradient boosting; expected to outperform Random Forest on structured data.

### 8.1 Theoretical framing

XGBoost is a gradient-boosted tree ensemble that fits trees sequentially, each new tree minimizing the residual error of the current ensemble. Compared to Random Forest, which averages independent trees, boosting *adds* trees, each correcting the previous ensemble's errors. This generally allows lower bias at the cost of higher variance — controlled by hyperparameters like `learning_rate`, `max_depth`, and various regularization terms.

XGBoost's hyperparameter space is large; the team used a **multi-step (sequential) grid search** rather than a single multidimensional search to keep computation tractable.

### 8.2 Multi-Step Hyperparameter Tuning

The tuning protocol follows a published recipe (cited in the notebook to Aarshay Jain's Analytics Vidhya guide):

1. **Step 0**: Choose initial `learning_rate` and `n_estimators`
2. **Step 1**: Tune `max_depth` and `min_child_weight` (controlling tree complexity)
3. **Step 2**: Tune `gamma` (minimum loss reduction for splitting)
4. **Step 3**: Tune `subsample` and `colsample_bytree` (stochastic subsampling)
5. **Step 4**: Tune `reg_alpha` (L1 regularization on leaf weights)
6. **Step 5**: Refine `reg_alpha`

At each step, only the parameters in question are searched; previously-tuned parameters are held fixed at their best-so-far values.

### 8.3 Experiment 1 — XGBoost with 1 feature, 2020 excluded

Final tuned parameters:

```
max_depth = 7, min_child_weight = 3, gamma = 0.4,
colsample_bytree = 0.7, subsample = 0.7, reg_alpha = 0.0
```

Test results:

```
Train MSE: 0.0330, Train MAE: 0.136
Test MSE:  0.0541, Test MAE: 0.167
```

Like Random Forest, XGBoost on the 1-feature input underperforms Linear Regression (0.054 vs. 0.028). Again, gradient boosting's additional capacity is unproductive on a near-linear three-feature input.

### 8.4 Experiment 2 — XGBoost with 11 features, 2020 excluded — **WINNING MULTIVARIATE MODEL**

Final tuned parameters:

```
learning_rate = 0.06, n_estimator = 40,
max_depth = 7, min_child_weight = 5,
gamma = 0.0, colsample_bytree = 0.9, subsample = 0.8,
reg_alpha = 0.06
```

Test results:

```
Train MSE: 0.106, Train MAE: 0.274
Test MSE:  0.0556, Test MAE: 0.188
```

This is the **best multi-feature result in the entire project**. Two notable features:

1. **Train MSE > Test MSE.** Unusual but explainable: heavy regularization (low `n_estimator`, `subsample = 0.8`, `colsample_bytree = 0.9`, plus L1 via `reg_alpha`) deliberately under-fits the training data to ensure good generalization.
2. **Test MSE is comparable to the single-feature Linear Regression** (0.056 vs. 0.028). XGBoost on 11 features matches the simple AR baseline, suggesting that the supplementary economic indicators provide ~50% as much predictive power as the autoregressive component when the model class is appropriately regularized.

### 8.5 Why XGBoost outperforms Random Forest on the 11-feature task

Three reasons:

1. **L1 regularization on leaf weights** (`reg_alpha`) effectively performs feature selection within the boosting process — irrelevant features get zero leaf weight.
2. **Sequential error-correction** allows later trees to focus on the residual structure left by earlier trees, rather than each tree independently re-discovering the dominant autoregressive signal.
3. **Heavy subsampling** (rows and columns) during tuning produced a much more conservative final model than Random Forest's defaults, fighting overfitting on the small dataset.

---

## 9. Model 4 — LSTM

**Role:** Test the deep-learning hypothesis that sequence-aware architectures can outperform structured-feature ML on time-series tasks.

### 9.1 Theoretical framing

A **Long Short-Term Memory** network is a recurrent neural network architecture that addresses the vanishing/exploding gradient problem of vanilla RNNs through three gating mechanisms (input, forget, output gates) that control information flow through a memory cell. LSTMs have been broadly successful on sequential data — language modeling, machine translation, time-series forecasting — when sufficient data is available.

The team cites Fischer & Krauss (2018), who use LSTM to predict directional movements of S&P 500 constituent stocks and find LSTM outperforms Random Forest, deep MLP, and logistic regression. The expectation, transferred to this project, is that LSTM should be competitive with or better than tree ensembles.

### 9.2 Implementation

Implemented in Keras (TensorFlow backend). Key architectural choices:

```python
model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]),
               return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(16))
model.add(Dropout(0.6))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
```

- **Two stacked LSTM layers** (32 then 16 units)
- **High dropout (0.6) after each layer** — explicitly aggressive regularization
- **MAE loss** (rather than MSE) — robust to outliers
- **Adam optimizer** — adaptive per-parameter learning rates
- **400 epochs**, batch size 128 (or 72 for some experiments)

Inputs are normalized to [0, 1] using `MinMaxScaler` (LSTM gates are sensitive to input scale).

### 9.3 Experiment 1 — 3-month window, 2020 included

```
Test MSE: 1.207, Test MAE: 0.930
```

Severely degraded by the 2020 outlier. The model has no mechanism for handling regime shift.

### 9.4 Experiment 2 — 3-month window, 2020 excluded — **PROBLEMATIC**

```
Test MSE: 0.489, Test MAE: 0.610
```

This is **the worst result among all multi-feature models tested in the 2020-excluded regime**. The reason, identified clearly by the team, is **overfitting**:

> "...despite setting high dropout probability after each LSTM layer, the model still suffers from serious overfitting problem. One reason could be that our dataset is monthly data which is much smaller than datasets in other applications of neural networks."

The training loss curve shows the train and validation losses diverge after ~50 epochs and never reconverge, with validation loss flattening around 0.05 while training loss continues to decrease.

### 9.5 Experiment 3 — 1-month window, 2020 excluded — **MUCH IMPROVED**

A creative diagnostic intervention: reduce the input window from 3 months to 1 month.

```python
model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
```

Note: a single LSTM layer, no dropout (which actually *hurt* performance here).

```
Test MSE: 0.067, Test MAE: 0.231
```

This is a 7× improvement over the 3-month window. The diagnosis: with a 3-month window of 11 features = 33 input features, the network has enough capacity to memorize the training set; with a 1-month window of 11 = 11 features, capacity is reduced to a level the data can support.

This experiment is a **clean illustration of the bias-variance tradeoff in neural architectures with small datasets**: more representational capacity is harmful when the training set cannot constrain the parameters.

### 9.6 Experiment 4 — 1-month window, 2020 included

```
Test MSE: 0.496, Test MAE: 0.341
```

Still substantially degraded by 2020 but **the LSTM with 1-month window is the second-best model on the 2020-included data after Linear Regression** — suggesting some robustness to the regime shift.

### 9.7 Multi-Step Forecasting Experiment

A separate experiment tested **multi-step forecasting** (predicting more than 1 month ahead from a 3-month input window):

| Configuration | Test RMSE |
|---|---|
| 3-month input → multi-step output, 2020 included | 0.776 |
| 3-month input → multi-step output, 2020 excluded | 0.245 |

The multi-step task is inherently harder (errors accumulate as predictions become inputs to subsequent predictions), so direct comparison to single-step is misleading. The 0.245 RMSE on 2020-excluded multi-step is, however, an interesting result — it suggests LSTM may shine on tasks the other models cannot perform.

### 9.8 Diagnosis and What Future Work Would Address

The team identifies the underlying problem clearly: **monthly data from 1959 to 2019 yields ~700 examples — orders of magnitude smaller than typical LSTM training corpora**. Two recommended remedies were noted but not implemented:

1. **L1/L2 regularization on the LSTM weights**, beyond dropout.
2. **Early stopping** based on validation loss.

A more aggressive remedy, not in the report but worth noting, would be transfer learning from a pre-trained time-series foundation model — though such models were nascent in 2020.

---

## 10. Model 5 — ARIMA (Comparative Baseline Only)

**Role:** Reference baseline from classical statistical time-series literature; *not directly comparable* to the supervised ML models.

### 10.1 Why ARIMA cannot be directly compared

Important methodological point that the team is careful to flag:

> "ARIMA simply finds the best fit to the time series by minimizing AIC or log likelihood for all the data in the training set... they have intrinsically different goals."

Specifically:
- The supervised models predict UNRATE one month ahead given 3 months of input. They are trained to minimize one-step-ahead MSE.
- ARIMA is fit to maximize the likelihood of the *entire* training series under a stochastic-process model. Forecasts come from iterating the fitted process, which (a) extrapolates from the last observation and (b) regresses toward the unconditional mean as the forecast horizon grows.

Comparing one-step MSE between the two model classes is not apples-to-apples.

### 10.2 Implementation

The team performed the standard ARIMA workflow:

1. **Stationarity check** via Augmented Dickey-Fuller test.
2. **First-order differencing** (`d = 1`) to remove trend.
3. **Seasonality check** — none observed for monthly unemployment (year-over-year and month-of-year variation are minor).
4. **Grid search** over `(p, q)` with `P = Q = 0` (non-seasonal):

```python
p = range(12); q = range(12); d = range(1, 2)
P = range(1); Q = range(1); D = range(1)
param_list = list(product(p, q, P, Q, d, D))
result_table = fitSARIMA(x_train, param_list, s=12)
```

The model selection criterion is AIC.

### 10.3 Result

The best fit is **ARIMA(5, 1, 9)** with **AIC = −436.11**:

```
Train MSE: 0.0265
Test MSE:  7.171
```

The test MSE is dominated by the 2020 spike (the test set in this evaluation includes 2020 by construction — the ARIMA notebook does not implement the 2020-excluded variant separately).

The team's qualitative assessment of ARIMA captures its core limitation:

> "It only models the mean of a time series, so the forecast values are around the mean of the series with decreasing variance, and finally converges to the mean, giving bad performance."

This is a textbook depiction of ARIMA's behavior: forecasts revert to the unconditional mean exponentially fast, leaving the model unable to capture even a few months of meaningful trend.

### 10.4 What ARIMA *is* useful for here

ARIMA's value in this report is not as a competitive forecaster but as a **demonstration of why economically-motivated supervised features matter**: even a sophisticated stochastic-process model trained on UNRATE alone cannot beat a simple linear regression on UNRATE_t-1. The autoregressive structure that linear regression captures with three weights is, for this time series, more useful than ARIMA's full residual-correlation modeling.

---

## 11. Comparative Analysis and Discussion

### 11.1 Headline Result Tables

**Test set: Jan 1959 – Dec 2019** (2020 excluded)

| Method | 1 feature MSE | 1 feature MAE | 11 features MSE | 11 features MAE |
|---|---|---|---|---|
| **Linear Regression** | **0.028** | **0.130** | 0.094 | 0.245 |
| Random Forest | 0.035 | 0.144 | 0.191 | 0.382 |
| XGBoost | 0.054 | 0.171 | **0.055** | **0.190** |
| LSTM (3-month window) | — | — | 0.489 | 0.610 |
| LSTM (1-month window) | — | — | 0.067 | 0.231 |

**Test set: Jan 1959 – Aug 2020** (2020 included)

| Method | 1 feature MSE | 1 feature MAE | 11 features MSE | 11 features MAE |
|---|---|---|---|---|
| Linear Regression | 0.938 | 0.250 | 0.836 | 0.365 |
| Random Forest | 0.842 | 0.261 | — | — |
| XGBoost | 0.892 | 0.280 | 0.933 | 0.285 |
| LSTM (3-month) | — | — | 1.151 | 0.902 |
| LSTM (1-month) | — | — | 0.496 | 0.341 |

### 11.2 The Three Substantive Findings

**Finding 1: Linear Regression with 1 feature wins overall.** On the 2020-excluded benchmark, plain OLS on UNRATE-only produces test MSE 0.028 — better than every other model in every configuration. This is not because Linear Regression is theoretically stronger but because:
- The task is single-step (vs. multi-step would amplify nonlinearity)
- Unemployment is overwhelmingly autoregressive (>95% importance on UNRATE_t-1)
- The dataset is small (~580 training points), constraining how much model capacity can be productively used

**Finding 2: XGBoost wins the multivariate benchmark.** When using all 11 features, XGBoost (0.055) beats Linear Regression (0.094) by a clear margin and approaches the 1-feature LR baseline (0.028). XGBoost's L1 regularization and sequential error-correction exploit the additional features' signal where unregularized OLS amplifies their noise.

**Finding 3: LSTM is the wrong tool here.** Despite theoretical promise, the LSTM with 3-month window (0.489) is the worst multivariate model. Reducing to 1-month window (0.067) recovers competitive performance, but the result remains worse than XGBoost. The fundamental issue is sample size: LSTMs need orders of magnitude more data than 700 monthly observations to extract their characteristic advantage.

### 11.3 Why the COVID-19 Test Reveals Robustness Differences

When the test set includes 2020, every model's MSE increases substantially — most gain a factor of 10–30×. But the *MAE* is more revealing because it weighs the single April-2020 outlier less heavily.

Under MAE, the ranking on 2020-included data is:
- **LSTM (1-month):** 0.341 ✓ — performs best
- **XGBoost (11 features):** 0.285 — also robust
- **Linear Regression (11 features):** 0.365
- **LSTM (3-month):** 0.902 ✗ — worst

The LSTM with 1-month window's robustness is interesting: its lower-capacity architecture happens to extrapolate more conservatively, making it less catastrophically wrong on the 2020 spike. XGBoost similarly benefits from heavy regularization. The deeper LSTM, which fits training-set patterns more aggressively, has internalized more "normal-times" structure that 2020 violates wholesale.

### 11.4 The Validation/Test Distribution Shift

A persistent theme across the project is that **validation MSE is consistently lower than test MSE**. For Random Forest on 11 features, validation MSE was ~0.005 but test MSE was 0.190 — a 38× gap. The cause is that the chronological holdout produces a validation set (2014–2017) and test set (2018–2019) drawn from genuinely different sub-distributions: validation period coincides with an unusually steady decline in unemployment, while the test period sees brief upticks in late 2018.

This is a fundamental difficulty in time-series ML that cross-validation cannot solve. **Multiple validation periods (rolling-origin or expanding-window CV) would partially mitigate the issue**, though the team's chosen single-fold strategy is acceptable for the project's scope.

### 11.5 The Bias-Variance Picture Across Model Families

| Model | Bias | Variance | Verdict on this problem |
|---|---|---|---|
| LR (1 feature) | Low (linear ≈ true relationship) | Very low (3 weights) | **Optimal** |
| LR (11 features) | Low | Moderate (33 weights, no regularization) | Slightly overfit |
| Random Forest | Medium | High (without regularization) | Severely overfit on 11 features |
| XGBoost | Low | Controlled (L1 + subsample) | **Optimal among 11-feature models** |
| LSTM (3-month) | Low | Very high (~1000s of weights) | Severely overfit |
| LSTM (1-month) | Low | High but reduced | Decent |

The picture is consistent: **on small-data, near-linear, single-step time series, model variance is the binding constraint, not bias**. Whatever architecture controls variance most effectively wins.

### 11.6 What Would Generalize and What Would Not

This project's specific findings — Linear Regression beats LSTM — should not be extrapolated naively to all macroeconomic forecasting contexts. The reasons LR wins here are:

1. **Single-step horizon.** Multi-step (predicting 6, 12, 24 months ahead) introduces nonlinear error compounding that LR cannot model.
2. **Stable monthly autocorrelation.** Unemployment rate is unusually smooth; intraday financial returns or daily case counts in epidemics would not exhibit this property.
3. **Small dataset size.** With 100,000+ observations rather than 730, LSTM's capacity becomes usable.
4. **Simple feature space.** With 100+ correlated indicators and proper regularization, more flexible models would likely beat LR.

The general lesson is the older one: **model selection should match task complexity, not aspiration**. A simple linear model is the right answer when the truth is approximately linear and data is scarce.

### 11.7 Methodological Strengths and Weaknesses of the Project

**Strengths:**
- Correct use of `PredefinedSplit` for chronological CV.
- Honest reporting of experiments that failed (the 129-feature LR baseline; the overfit 3-month LSTM).
- Two parallel feature regimes (1 vs. 11) that isolate the contribution of exogenous indicators.
- Test of robustness under regime shift (with/without 2020).
- Custom `Window_generator` class that cleanly integrates with scikit-learn.

**Weaknesses:**
- **Single validation fold.** Rolling-origin CV would give more stable hyperparameter estimates.
- **No early stopping in LSTM training.** The team identified this in future work.
- **No regularized linear baseline (Lasso/Ridge).** A Lasso on 11 features would likely match the 1-feature OLS by automatically zeroing out the weak features — a methodologically tighter result than the unregularized 11-feature OLS.
- **Inconsistent target scaling between models.** The Random Forest and XGBoost notebooks use the original UNRATE scale, while the LSTM uses MinMax-scaled data; this is fine for individual models but complicates direct error comparison.
- **Seasonal-differencing experiment was exploratory only.** The R² benchmark on lag-12 features was promising but not pursued into a comparable test-MSE evaluation.

---

## 12. Conclusions and Future Work

### 12.1 Conclusions

This project concludes that, for the specific task of one-month-ahead unemployment rate forecasting using FRED-MD data:

1. **Plain Linear Regression on the previous 3 months of UNRATE is the strongest single-feature model**, achieving 0.028 test MSE on 2019 data — better than Random Forest, XGBoost, and LSTM in any configuration.

2. **XGBoost is the strongest multi-feature model** at test MSE 0.055, approaching the 1-feature LR baseline. It is the model of choice when interpretability matters less than the ability to incorporate exogenous indicators.

3. **LSTM, despite theoretical promise, underperforms classical ML** on this small monthly dataset. A reduced 1-month-window LSTM is competitive but does not surpass XGBoost.

4. **All models degrade catastrophically on COVID-19 data** (April 2020 unemployment spike), with the relative ranking of robustness being LSTM (1-month) > XGBoost > LR > LSTM (3-month) > RF in MAE terms.

5. **ARIMA, while traditional, is not a competitive forecaster** here — its forecast structure (mean-reversion with declining variance) is mismatched to the trend and persistence of unemployment.

The deeper takeaway: **on small-data, single-step, near-linear time-series tasks, simpler models should be the default**. Capacity is only beneficial when the problem structure rewards it, and forcing capacity onto a problem that doesn't reward it produces overfit predictions in proportion to the model's flexibility.

### 12.2 Future Work — Acknowledged in the Report

The team explicitly identifies several directions:

1. **Multi-step forecasting** — predicting `n` months ahead. This is genuinely harder and would likely change model rankings in favor of LSTM or other sequence-aware models.
2. **Longer input windows** — using 6 or 12 months of history rather than 3.
3. **Variable-length time windows.**
4. **Searching for better leading indicators** — economic indicators whose changes precede unemployment changes, rather than coincide with them. Examples: yield-curve inversion, manufacturing PMI, initial jobless claims (already in the 11-feature set), consumer-confidence indices.
5. **Implementing L1/L2 regularization on the LSTM and early stopping** to address the documented overfitting.

### 12.3 Future Work — Not in the Report But Implied

Several methodological extensions would tighten the analysis:

6. **Regularized linear models** (Lasso, Ridge, ElasticNet) on the 11-feature set, to test whether OLS's shortfall is due to feature noise or fundamental linearity-assumption failure.
7. **Rolling-origin cross-validation** for hyperparameter selection.
8. **Confidence intervals on test MSE** via bootstrap resampling of the test set residuals.
9. **Transformer-based time-series models** (PatchTST, Informer, foundation models like TimesFM, Chronos) — these have largely supplanted LSTM for time-series in subsequent literature and would likely outperform LSTM on this small dataset due to pre-training transfer.
10. **Calibration analysis** — for any of these models, do the prediction errors satisfy approximately normal residuals? A QQ-plot inspection would reveal whether the MSE-minimizing models are also probabilistically well-calibrated.

---

## 13. Appendix — File Inventory and Indicator Descriptions

### 13.1 File Inventory

| File | Role | Notebook section |
|---|---|---|
| `Stat451_Final_Project_Part1.ipynb` | EDA, feature selection, sliding-window class, LR / RF / XGBoost / ARIMA implementations | Sections 3–8, 10 |
| `STAT451UnemploymentForecast.ipynb` | LSTM implementations with all four window/regime configurations | Section 9 |
| `Econ_predictors_2020-08.csv` | Raw FRED-MD data, 740 rows × 130 columns | All |
| `clean_econ_predictors.csv`, `clean_econ_predictors1.csv`, `clean_econ_predictors2.csv` | Cleaned/reduced subsets used by LSTM notebook | LSTM only |
| `seasonally_adjusted.csv`, `lags_12months_features.csv` | Intermediate artifacts from the exploratory seasonal-differencing experiment | Random Forest section |
| `451_Final_Project.pdf` | Final written report (8 pages) | All |

### 13.2 The 11 Selected Economic Indicators

The 11 indicators used as features (10 + UNRATE) with their FRED-MD codes and meanings:

| Code | Description |
|---|---|
| **UNRATE** | Civilian Unemployment Rate (target variable, also used as feature) |
| **CUMFNS** | Capacity Utilization: Manufacturing |
| **BAAFFM** | Moody's Baa Corporate Bond Yield minus Federal Funds Rate (credit spread) |
| **CLAIMSx** | Initial unemployment-insurance claims |
| **HWIURATIO** | Help Wanted Index / Number of Unemployed (labor-market tightness) |
| **UEMPLT5** | Civilians Unemployed Less Than 5 Weeks |
| **UEMP5TO14** | Civilians Unemployed 5 to 14 Weeks |
| **UEMP15OV** | Civilians Unemployed 15 Weeks and Over |
| **UEMP15T26** | Civilians Unemployed 15 to 26 Weeks |
| **UEMP27OV** | Civilians Unemployed 27 Weeks and Over |
| **CES1021000001** | All Employees: Mining and Logging: Mining |

### 13.3 Software Stack

- **Python 3** with NumPy, pandas, matplotlib, seaborn
- **scikit-learn**: `LinearRegression`, `RandomForestRegressor`, `GridSearchCV`, `RandomizedSearchCV`, `PredefinedSplit`, `StandardScaler`, `MinMaxScaler`
- **XGBoost**: `XGBRegressor`
- **Keras / TensorFlow**: LSTM, Dropout, Dense, Adam optimizer
- **statsmodels**: SARIMAX, ADF stationarity test, ACF/PACF
- **Environments**: Jupyter Notebook (local), Google Colab, DeepNote

---

*End of report.*
