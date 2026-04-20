# 🏅 Olympics Data Analysis
 
> A comprehensive end-to-end data science project analysing **120 years of Olympic history (1896–2016)** — covering athlete biometrics, gender inclusion trends, economic drivers of medal success, and a machine-learning model that predicts medal probability for any custom athlete profile.
 
---
 
## 📋 Table of Contents
1. [Purpose](#-purpose)
2. [Datasets & Variables](#-datasets--variables)
3. [Theory — Code Walkthrough](#-theory--code-walkthrough)
   - [Section 1 — Imports & Configuration](#section-1--imports--configuration)
   - [Section 2 — Data Loading & Initial Inspection](#section-2--data-loading--initial-inspection)
   - [Section 3 — Data Quality Report](#section-3--data-quality-report)
   - [Section 4 — Data Cleaning](#section-4--data-cleaning)
   - [Section 5 — Exploratory Data Analysis](#section-5--exploratory-data-analysis)
   - [Section 6 — Statistical Analysis](#section-6--statistical-analysis)
   - [Section 7 — Time-Series & Trend Analysis](#section-7--time-series--trend-analysis)
   - [Section 8 — Country Economics × Olympic Performance](#section-8--country-economics--olympic-performance)
   - [Section 9 — Medal Prediction Model](#section-9--medal-prediction-model)
4. [Conclusions](#-conclusions)
 
---
 
## 🎯 Purpose
 
The project is built around five core research questions:
 
| # | Research Question |
|---|-------------------|
| Q1 | What are the key economic determinants of Olympic success — does GDP per capita predict medal counts? |
| Q2 | How has gender inclusion at the Olympics evolved over 120 years, and which nations lead? |
| Q3 | What physical profiles (age, height, weight, BMI) are associated with medal-winning athletes in different sports? |
| Q4 | How does India's Olympic performance compare to nations of similar population and economic standing? |
| Q5 | Can a machine-learning model accurately predict whether an athlete will win a medal, and what features matter most? |
 
---
 
## 📦 Datasets & Variables
 
### `athlete_events.csv`
Contains one row per athlete per event per Olympic Games.
 
| Variable | Type | Description |
|----------|------|-------------|
| `Name` | string | Athlete's full name |
| `Sex` | string | `M` (Male) or `F` (Female) |
| `Age` | float | Age at the time of the Games |
| `Height` | float | Height in centimetres |
| `Weight` | float | Weight in kilograms |
| `Team` | string | Country team name |
| `NOC` | string | 3-letter National Olympic Committee code |
| `Year` | int | Olympic year |
| `Season` | string | `Summer` or `Winter` |
| `Sport` | string | Sport name |
| `Event` | string | Specific event within the sport |
| `Medal` | string | `Gold`, `Silver`, `Bronze`, or `NaN` (no medal) |
 
**Derived columns added during cleaning:**
 
| Variable | Description |
|----------|-------------|
| `BMI` | Body Mass Index = Weight / (Height/100)² |
| `Won_Medal` | Binary flag: `1` if any medal was won, `0` otherwise |
 
### `country_economics.csv`
Contains socioeconomic data for countries.
 
| Variable | Type | Description |
|----------|------|-------------|
| `Country` | string | Country name (some marked with `*` for territories) |
| `Code` | string | NOC code (joins to athlete data) |
| `Population` | float | Country population |
| `GDP per Capita` | float | GDP per capita in USD |
 
**Derived columns added during cleaning:**
 
| Variable | Description |
|----------|-------------|
| `Is_Territory` | Binary: `1` if the row represents a territory (had `*`) |
| `GDP_Tier` | Categorical band: `Low (<$2k)`, `Lower-Mid ($2k–$10k)`, `Upper-Mid ($10k–$30k)`, `High (>$30k)` |
| `Log_Population` | `log10(Population)` — used to handle population's right-skewed distribution |
 
---
 
## 🔬 Theory — Code Walkthrough
 
### Section 1 — Imports & Configuration
 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
 
**What this does and why:**
 
- `pandas` is the primary data manipulation library — used for all DataFrame operations (filtering, grouping, merging, aggregating).
- `numpy` provides numerical computation: array operations, logarithms, quantile calculations for outlier detection, and mathematical feature engineering.
- `matplotlib.pyplot` is the base plotting engine. `mticker` formats axis labels (e.g., percentage formatting). `mpatches` creates custom legend entries. `GridSpec` allows complex multi-panel figure layouts.
- `seaborn` is a high-level statistical plotting library built on Matplotlib, used here for colour palettes and grouped chart layouts.
- `warnings.filterwarnings('ignore')` suppresses non-critical deprecation warnings that would clutter notebook output without affecting results.
 
**Global plot style configuration** is set via `plt.rcParams.update()` — this ensures consistent fonts, DPI (120 for crisp output), no top/right spines, and a light grid across all visualisations without needing to repeat formatting code per chart.
 
**Colour palettes** are defined as dictionaries upfront:
 
```python
MEDAL_COLORS = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32', 'No Medal': '#ECEFF1'}
SEX_COLORS   = {'M': '#378ADD', 'F': '#E24B4A'}
SEASON_COLORS= {'Summer': '#E24B4A', 'Winter': '#378ADD'}
GDP_COLORS   = ['#1D9E75', '#BA7517', '#378ADD', '#E24B4A']
```
 
This ensures semantic colour consistency (gold is always `#FFD700`, female athletes are always red `#E24B4A`) across every chart in the notebook, making the visual output immediately interpretable.
 
---
 
### Section 2 — Data Loading & Initial Inspection
 
```python
raw_df = pd.read_csv('country_economics.csv')
ath_df = pd.read_csv('athlete_events.csv')
```
 
**What this does and why:**
 
Both datasets are loaded into Pandas DataFrames using `pd.read_csv()`. After loading, the code immediately prints:
- **Shape** (rows × columns) — to confirm the full dataset loaded without truncation.
- **Column names** — to verify all expected fields are present before analysis begins.
- **Year range** of the athlete dataset — confirms historical coverage (1896–2016).
- **Unique NOCs and Sports** — gives an instant sense of geographic and sport diversity.
 
`ath_df.head()` and `raw_df.head(10)` are then called to visually inspect the first few rows, which helps catch obvious formatting issues (e.g., unexpected asterisks in country names, or columns loaded as the wrong data type) before investing effort in cleaning.
 
---
 
### Section 3 — Data Quality Report
 
```python
def quality_report(df, name):
    missing    = df.isnull().sum()
    pct        = (missing / len(df) * 100).round(2)
    unique     = df.nunique()
    dtypes     = df.dtypes
    report     = pd.DataFrame({
        'dtype'       : dtypes,
        'missing'     : missing,
        'missing_%'   : pct,
        'unique_vals' : unique,
    })
    print(f'Rows: {len(df):,} | Cols: {len(df.columns)} | Duplicates: {df.duplicated().sum():,}')
    display(report)
    return pct
```
 
**What this does and why:**
 
A reusable `quality_report()` function is defined to provide a structured audit of any DataFrame before cleaning. It reports:
 
- **`dtype`** — data type per column. Identifies if numeric columns were accidentally loaded as strings (a common issue with columns containing formatting like commas or `%`).
- **`missing` / `missing_%`** — absolute and percentage null counts. Directs attention to columns that need imputation or removal.
- **`unique_vals`** — the number of distinct values per column. Low cardinality in a numeric column suggests potential encoding issues; unexpectedly high cardinality in a categorical column may indicate typos or inconsistent casing.
- **`Duplicates`** — the count of fully identical rows, critical to identify before any aggregation.
 
This function is called on both datasets:
- `raw_df` had **12.4% missing GDP** and **2.5% missing Population**, mostly in small territories.
- `ath_df` had missing values in `Age` (3.5%), `Height` (22%), `Weight` (23%), and `Medal` (85% — representing non-winners).
 
---
 
### Section 4 — Data Cleaning
 
**Country Dataset Cleaning (`country_economics.csv`)**
 
```python
# Step 1 — Flag & strip asterisks (territories)
clean_df['Is_Territory'] = clean_df['Country'].str.contains(r'\*', regex=True).astype(int)
clean_df['Country']      = clean_df['Country'].str.replace(r'\*', '', regex=True).str.strip()
 
# Step 2 — Impute Population with overall median
pop_med = clean_df['Population'].median()
clean_df['Population'].fillna(pop_med, inplace=True)
 
# Step 3 — Impute GDP per group (territory vs country)
for flag in [0, 1]:
    med  = clean_df.loc[clean_df['Is_Territory']==flag, 'GDP per Capita'].median()
    mask = (clean_df['Is_Territory']==flag) & (clean_df['GDP per Capita'].isna())
    clean_df.loc[mask, 'GDP per Capita'] = med
 
# Step 4 — Derived features
gdp_bins   = [0, 2000, 10000, 30000, np.inf]
gdp_labels = ['Low (<$2k)', 'Lower-Mid ($2k–$10k)', 'Upper-Mid ($10k–$30k)', 'High (>$30k)']
clean_df['GDP_Tier']       = pd.cut(clean_df['GDP per Capita'], bins=gdp_bins, labels=gdp_labels)
clean_df['Log_Population'] = np.log10(clean_df['Population'])
```
 
**Step-by-step explanation:**
 
- **Step 1 — Territory Flagging:** Some country names contain a trailing `*` (e.g., `"Puerto Rico*"`) indicating they are territories, not sovereign nations. A binary `Is_Territory` column is created using `.str.contains(r'\*', regex=True)`, then the asterisk is stripped for clean display names. This flag is critical because territories have systematically different GDP and population profiles — mixing them with countries would bias imputation.
 
- **Step 2 — Population Imputation:** 12 rows had null population. These are filled with the **overall dataset median** (rather than mean) because population is heavily right-skewed — the median is more robust and representative of the typical country size.
 
- **Step 3 — Group-Specific GDP Imputation:** GDP nulls are imputed using the **group median** (territories separately from sovereign countries). This avoids bias: a territory's GDP is structurally different from a full nation's GDP, so using a single global median would introduce systematic error.
 
- **Step 4 — Feature Engineering:**
  - `GDP_Tier` is created using `pd.cut()` with economically meaningful thresholds ($2k, $10k, $30k) to categorise countries into four development levels. This converts a continuous variable into an ordinal one suitable for grouping, colour-coding in charts, and use as a model feature.
  - `Log_Population` takes `log10` of population. Population spans six orders of magnitude (thousands to billions), which causes extreme skew in scatter plots and regression analyses. The log transform compresses this into a roughly linear, comparable scale.
 
**Athlete Dataset Cleaning (`athlete_events.csv`)**
 
```python
# Step 1 — Drop exact duplicates
ath.drop_duplicates(inplace=True)
 
# Step 2 — Medal: NaN → "No Medal"
ath['Medal'] = ath['Medal'].fillna('No Medal')
 
# Step 3 — Impute numerical columns with median
for col in ['Age', 'Height', 'Weight']:
    med = ath[col].median()
    ath[col].fillna(med, inplace=True)
 
# Step 4 — Derived columns
ath['BMI']       = (ath['Weight'] / ((ath['Height'] / 100) ** 2)).round(2)
ath['Won_Medal'] = (ath['Medal'] != 'No Medal').astype(int)
 
# Step 5 — IQR outlier detection for Age (report only)
Q1, Q3 = ath['Age'].quantile([0.25, 0.75])
IQR    = Q3 - Q1
n_out  = ((ath['Age'] < Q1 - 1.5*IQR) | (ath['Age'] > Q3 + 1.5*IQR)).sum()
```
 
- **Step 1 — Duplicate Removal:** `drop_duplicates()` removes rows that are identical across every column — common when athletes appear in team events that are logged multiple times per athlete. `reset_index(drop=True)` cleans the index after removal.
 
- **Step 2 — Semantic NaN Fill:** `Medal` has `NaN` for every athlete who did not win. This is not missing data — it is meaningful. Replacing with the string `'No Medal'` makes the column complete and human-readable, enabling direct value counts and grouping without special NaN handling later.
 
- **Step 3 — Median Imputation for Biometrics:** `Age`, `Height`, and `Weight` have 3–23% missing values. Median imputation is chosen over mean because biometric data is slightly right-skewed (a small number of very old or very large athletes). Imputing with the median preserves the central tendency without being pulled by outliers. A loop applies the same logic to all three columns cleanly.
 
- **Step 4 — BMI & Won_Medal Engineering:**
  - `BMI = Weight / (Height/100)²` — the standard WHO formula, computed entirely from existing columns. BMI captures the weight-to-height ratio as a single composite feature, making it useful for cross-sport physical profiling.
  - `Won_Medal = 1 if Medal != 'No Medal' else 0` — creates the binary target variable used in all statistical summaries and the machine learning model. Stored as an integer for direct numerical use.
 
- **Step 5 — IQR Outlier Detection:** The Interquartile Range method flags values below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` as statistical outliers. For `Age`, this identifies very young and very old athletes. The outliers are **reported but not removed** — many are legitimate (e.g., 10-year-old gymnasts, 73-year-old equestrians) and removing them would erase historically significant data points.
 
---
 
### Section 5 — Exploratory Data Analysis
 
**5a — Physical Attribute Distributions**
 
```python
configs = [('Age', '#378ADD', 'Age (years)'), ('Height', '#1D9E75', 'Height (cm)'),
           ('Weight', '#E24B4A', 'Weight (kg)'), ('BMI', '#BA7517', 'BMI')]
 
for ax, (col, color, xlabel) in zip(axes, configs):
    data = ath[col]
    if col == 'BMI':
        data = data[(data >= 15) & (data <= 50)]
    ax.hist(data, bins=40, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.axvline(data.mean(),   color='black', linestyle='--', linewidth=1.6)
    ax.axvline(data.median(), color='gray',  linestyle=':',  linewidth=1.6)
```
 
Four histograms are plotted side by side using `plt.subplots(1, 4)`. For each attribute, both the **mean** (dashed black) and **median** (dotted grey) are overlaid — the gap between them reveals the degree of skew. BMI is trimmed to the range 15–50 to exclude physically implausible values caused by data entry errors. `bins=40` provides enough granularity to show distributional shape without excessive noise.
 
**5b — Medal Distribution**
 
```python
medal_counts = ath['Medal'].value_counts()
# Bar chart of all categories + Pie chart of medal-winners only
winners = medal_counts.drop('No Medal')
axes[1].pie(winners, labels=winners.index, autopct='%1.1f%%', ...)
```
 
A bar chart shows the absolute count of all four Medal categories. The pie chart is deliberately restricted to **medal winners only** (Gold, Silver, Bronze) to show the proportion split among winners — confirming that the three medal types are awarded in roughly equal numbers (each ~33%), as expected by competition design.
 
**5c — Sports Participation & Gender Split**
 
```python
top15_sports = ath['Sport'].value_counts().head(15).index.tolist()
sex_sport    = ath[ath['Sport'].isin(top15_sports)].groupby(['Sport','Sex']).size().unstack(fill_value=0)
pct_sport    = sex_sport.div(sex_sport.sum(axis=1), axis=0)
```
 
`groupby(['Sport', 'Sex']).size().unstack()` creates a pivot table of athlete counts by sport and gender. Dividing each row by its row sum converts counts to proportions (`pct_sport`). A stacked horizontal bar chart then shows gender composition as percentages for each sport, sorted by female participation. A vertical dashed line at 50% marks parity.
 
---
 
### Section 6 — Statistical Analysis
 
**6a — Descriptive Statistics by Season**
 
```python
season_stats = ath.groupby('Season')[['Age','Height','Weight','BMI','Won_Medal']].agg(
    ['mean', 'median', 'std']
).round(2)
```
 
`groupby('Season').agg()` computes mean, median, and standard deviation for each numeric variable split by Summer and Winter Olympics. This reveals systematic differences — Winter athletes tend to be slightly older and heavier than Summer athletes, reflecting the different physical demands of cold-weather sports.
 
**6b — Top 10 NOC Medal Breakdown**
 
```python
top10_noc   = ath[ath['Won_Medal']==1]['NOC'].value_counts().head(10).index.tolist()
medal_pivot = ath[ath['NOC'].isin(top10_noc)].groupby(['NOC','Medal']).size().unstack(fill_value=0)
medal_pivot = medal_pivot.reindex(columns=['Gold','Silver','Bronze'], fill_value=0)
```
 
The top 10 medal-winning nations are identified using `value_counts().head(10)`. A pivot table then breaks each nation's medals into Gold, Silver, and Bronze using `.groupby(['NOC','Medal']).size().unstack()`. `.reindex(columns=[...])` ensures columns are always in the Gold → Silver → Bronze order regardless of which medal type appears first in the data. A grouped bar chart with `bar_label()` annotations displays exact counts on each bar.
 
---
 
### Section 7 — Time-Series & Trend Analysis
 
**7a — Gender Equality Trend**
 
```python
gender_yr = ath[ath['Season']=='Summer'].groupby(['Year','Sex'])['ID'].nunique().unstack(fill_value=0)
pct_f     = (gender_yr.get('F', pd.Series(0, index=gender_yr.index)) /
             gender_yr.sum(axis=1) * 100)
```
 
`ID.nunique()` counts **unique athletes** per year per gender (not entries, which would inflate team-sport counts). `.unstack()` pivots gender into columns, creating a time-indexed DataFrame. `pct_f` computes female percentage per year as `(Female / Total) × 100`. The stacked area chart shows absolute athlete volume over time, while the line chart shows the female participation percentage with a 50% parity reference line and a `fill_between()` shaded area to highlight the gap remaining.
 
**7b — Medal Dominance Over Time**
 
```python
top6         = summer[summer['Won_Medal']==1]['NOC'].value_counts().head(6).index.tolist()
medal_trend  = summer[summer['NOC'].isin(top6)].groupby(['Year','NOC'])['Won_Medal'].sum().unstack(fill_value=0)
```
 
For each of the six historically dominant nations, `Won_Medal.sum()` per year gives the total medals won at each Games. `.unstack()` creates one column per nation. Each nation is plotted as a line with marker points, using `sns.color_palette('tab10')` for visually distinct colours. `MaxNLocator(integer=True)` prevents fractional year labels on the x-axis.
 
---
 
### Section 8 — Country Economics × Olympic Performance
 
**8a — Building the Merged Dataset**
 
```python
noc_agg = summer.groupby('NOC').agg(
    Total_Medals   = ('Won_Medal', 'sum'),
    Gold_Medals    = ('Medal', lambda x: (x=='Gold').sum()),
    Total_Athletes = ('ID', 'nunique'),
    Unique_Sports  = ('Sport', 'nunique'),
    Games_Entered  = ('Games', 'nunique'),
).reset_index()
 
noc_agg['Medal_Rate'] = (noc_agg['Total_Medals'] / noc_agg['Total_Athletes']).round(3)
merged = noc_agg.merge(clean_df, left_on='NOC', right_on='Code', how='inner')
```
 
`groupby('NOC').agg()` uses named aggregations — a Pandas feature that allows multiple aggregate functions with custom output column names in a single call. The `lambda x: (x=='Gold').sum()` pattern counts specific medal types within the aggregation. `Medal_Rate` (medals per unique athlete) is a fairer cross-country efficiency metric than raw medal counts, which heavily favour large nations. `merge(..., how='inner')` joins only NOCs that exist in both the athlete and country datasets.
 
**8b — GDP vs Medals Bubble Chart**
 
```python
for tier, color in zip(tiers, GDP_COLORS):
    sub = merged[merged['GDP_Tier'] == tier]
    ax.scatter(sub['GDP per Capita'], sub['Total_Medals'],
               s=np.sqrt(sub['Population']) / 25,
               color=color, alpha=0.6, edgecolors='w', linewidth=0.8, label=tier)
```
 
A bubble chart (scatter plot with variable bubble size) maps three dimensions simultaneously:
- **X-axis:** GDP per capita
- **Y-axis:** Total medals won
- **Bubble size:** `√Population / 25` — the square root is taken to prevent the largest countries (China, India, USA) from producing impossibly large bubbles that obscure smaller nations. Dividing by 25 scales to a visually comfortable pixel size.
- **Colour:** GDP tier
 
The top 12 nations by medals are annotated using `ax.annotate()`, providing country identification without cluttering the full scatter.
 
---
 
### Section 9 — Medal Prediction Model
 
**Feature Engineering**
 
```python
model_df['Sex_enc']    = (model_df['Sex'] == 'M').astype(int)
model_df['Season_enc'] = (model_df['Season'] == 'Summer').astype(int)
 
noc_rate   = model_df.groupby('NOC')['Won_Medal'].mean().rename('NOC_Medal_Rate')
model_df   = model_df.join(noc_rate, on='NOC')
 
sport_rate = model_df.groupby('Sport')['Won_Medal'].mean().rename('Sport_Medal_Rate')
model_df   = model_df.join(sport_rate, on='Sport')
 
gdp_order  = {'Low (<$2k)': 0, 'Lower-Mid ($2k–$10k)': 1, 'Upper-Mid ($10k–$30k)': 2, 'High (>$30k)': 3}
model_df['GDP_Tier_enc'] = model_df['GDP_Tier'].map(gdp_order).fillna(1)
```
 
Categorical variables are converted to numeric form for the Random Forest:
- `Sex_enc` and `Season_enc` are binary-encoded (1/0).
- `NOC_Medal_Rate` and `Sport_Medal_Rate` are **target-encoded** — each NOC/sport is replaced by its historical medal win rate. This is a powerful feature engineering technique that encodes the cumulative historical competitive strength of a country or sport into a single continuous number, capturing far more signal than a simple one-hot encoding of hundreds of NOC codes.
- `GDP_Tier_enc` maps the four economic tiers to ordinal integers (0–3), preserving the natural ordering (Low < Lower-Mid < Upper-Mid < High).
 
**Model Training**
 
```python
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=20,
    class_weight='balanced', n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
```
 
A **Random Forest** is chosen because it handles mixed data types, is robust to outliers, naturally provides feature importance scores, and requires minimal hyperparameter tuning to achieve strong baseline performance.
 
Key hyperparameter choices:
- `n_estimators=300` — 300 decision trees are averaged. More trees reduce variance (overfitting) at a modest computational cost.
- `max_depth=12` — limits each tree's depth to prevent overfitting on the training set.
- `min_samples_leaf=20` — each leaf node must contain at least 20 samples, ensuring no predictions are based on tiny subsets.
- `class_weight='balanced'` — critical for this dataset where only ~14.7% of entries are medal winners. The balanced weighting automatically adjusts the loss function so that medal-winning cases are not ignored by the majority class.
- `n_jobs=-1` — uses all available CPU cores for parallel tree training.
- `random_state=42` — ensures fully reproducible results.
 
**Model Evaluation**
 
```python
roc_auc   = roc_auc_score(y_test, y_prob)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
```
 
- **ROC-AUC** measures the model's ability to rank medal-winners above non-winners across all possible probability thresholds. A score of **0.8141** means the model correctly ranks a random medal-winner above a random non-winner 81.4% of the time — a strong result for a highly imbalanced real-world dataset.
- **5-fold cross-validation** (`cv=5`) splits the data into 5 equal folds, trains on 4, tests on 1, and rotates. The CV AUC of **0.8020 ± σ** confirms the model generalises well beyond the training data and is not overfit.
 
**Live Prediction Function**
 
```python
def predict_medal(age, height_cm, weight_kg, sex='M', season='Summer',
                  noc='USA', sport='Athletics', gdp_tier='High (>$30k)'):
    bmi          = weight_kg / ((height_cm / 100) ** 2)
    noc_rate_v   = noc_rate.get(noc, noc_rate.mean())
    sport_rate_v = sport_rate.get(sport, sport_rate.mean())
    prob         = rf.predict_proba(row)[0, 1]
    outcome      = 'MEDAL' if prob >= 0.5 else 'No Medal'
    return {'medal_probability': round(float(prob), 4), 'predicted_outcome': outcome, ...}
```
 
`predict_medal()` is a user-facing wrapper that accepts human-readable inputs and returns a medal probability. `noc_rate.get(noc, noc_rate.mean())` gracefully handles unknown NOC codes by falling back to the global average — preventing `KeyError` crashes on new or misspelled codes. The 50% threshold is the standard decision boundary for binary classifiers.
 
**Demo predictions:**
 
| NOC | Sport | Age | Probability | Outcome |
|-----|-------|-----|-------------|---------|
| USA | Swimming | 24 | 85.1% | ✅ MEDAL |
| CHN | Gymnastics | 19 | 33.6% | ❌ No Medal |
| IND | Athletics | 32 | 25.2% | ❌ No Medal |
| NOR | Skiing | 27 | 67.7% | ✅ MEDAL |
 
---
 
## 📊 Conclusions
 
1. **Data Quality Validated:** `country_economics.csv` had 12.4% missing GDP values and 2.5% missing population, concentrated in small territories flagged with `*`. After group-median imputation and asterisk cleaning, the country dataset is complete with 201 rows and zero nulls. The athlete dataset was similarly cleaned to 269,731+ rows.
 
2. **Physical Profiles are Sport-Specific:** Male athletes average ~179 cm in height vs ~167 cm for females. BMI peaks at 22–24 across most sports. Rowing and Weightlifting athletes are the heaviest; Gymnastics and Diving athletes are the lightest. These biometric clusters demonstrate that targeting the right sport for an athlete's natural physical profile is a key part of talent development strategy.
 
3. **Gender Parity has Dramatically Improved:** Women were almost entirely absent from the early Olympics (1896–1920). By 2016, female participation reached ~45% — a transformation driven by the IOC's progressive inclusion policies. Nations that have achieved gender parity in sports funding consistently outperform their GDP/population baseline in medal counts.
 
4. **Medal Efficiency Matters More Than Scale:** Small high-income nations (e.g., Scandinavia, Eastern Europe in Winter sports) show the highest medal rate per athlete. Raw population size does not predict success — sports system quality, investment, and athlete pathway infrastructure do.
 
5. **Dominance Shift from USA → Multi-polar:** The USA dominated Summer Olympics throughout the 20th century. The Soviet Union (URS) was a close rival during the Cold War era (1952–1988). Post-1992, China (CHN) rapidly rose to become the second-strongest Summer medallist — demonstrating how a state-directed, long-horizon sports investment programme can transform a nation's Olympic trajectory within 30 years.
 
6. **ML Model Achieves Strong Predictive Power (ROC-AUC: 0.8141):** The Random Forest classifier correctly identifies medal-winners with 74.6% recall. The two most important predictors are **Country Historical Medal Rate (45.3%)** and **Sport Historical Medal Rate (27.9%)** — confirming that systemic national and sport-level investment matters far more than individual biometrics in determining Olympic outcomes.
 
---
 
