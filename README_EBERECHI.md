# Machine Learning Workflow: Educational Data Visualization & Performance Modeling

This repository explores the role of data visualization across the machine learning lifecycle using a comprehensive dataset of 5,000 U.S. educational institutions. Through two structured Jupyter notebooks, the project demonstrates how visual analytics bridges data auditing, strategic resource allocation, multicollinearity screening, supervised classification diagnostics, and unsupervised dimensionality reduction. Together, these workflows showcase how clear visual representations transform raw educational metrics into actionable analytical insights and interpretable machine learning models.

---

## Disclaimer

> [!WARNING]
> **Synthetic Dataset Notice:** The dataset analyzed in this repository (`us_education_system_dataset.csv`) is synthetic and designed solely for instructional simulations. It does **not** represent verified, real-world operational records of actual United States schools or school districts.
>
> - **Educational Use Only:** This analysis is conducted strictly for academic and training purposes within the AltSchool Africa Data Science curriculum. It must not be used to inform real-world public policy, municipal budgeting, school management, or financial investment decisions.
> - **Data Limitations:** Statistical findings, group differences, and model outputs are constrained by the assumptions, size, and generative distributions of the underlying synthetic dataset.

---

## Dataset Source & Overview

- **Dataset Name:** `us_education_system_dataset.csv`
- **Source & Provenance:** The dataset is aligned with the reporting conventions and variable standards of the [U.S. National Center for Education Statistics (NCES)](https://nces.ed.gov/), specifically drawing inspiration from the NCES Common Core of Data (CCD) and the Integrated Postsecondary Education Data System (IPEDS). It was curated as a standardized benchmark for AltSchool Africa's Machine Learning Workflow module.
- **Dimensions:** 5,000 institution records (rows) and 35 variables (columns).
- **Core Column Categories:**
  - **Identifiers & Geography:** `institution_id`, `institution_name`, `state_code`, `state_name`, `county`, `nces_locale` (e.g., City, Suburban, Town, Rural).
  - **Governance & Structure:** `level` (Elementary, Middle, High School, K-12, 2-Year College, 4-Year University), `control` (Public, Public Charter, Private Non-Profit, Private Religious, Private For-Profit), `title_i_status`.
  - **Staffing & Enrollment:** `total_enrollment`, `number_of_teachers`, `student_teacher_ratio`.
  - **Demographics & Equity:** `percent_free_reduced_lunch`, racial demographic shares (`racial_white_pct`, `racial_black_pct`, `racial_hispanic_pct`, etc.), `special_education_pct`, `english_learner_pct`.
  - **Finance:** `per_pupil_expenditure_usd`, `federal_funding_pct`, `state_funding_pct`, `local_funding_pct`.
  - **Academic Outcomes & Technology:** `math_proficiency_pct`, `reading_proficiency_pct`, `graduation_rate_pct`, `chronic_absenteeism_pct`, `avg_sat_score`, `avg_act_score`, `ap_ib_participation_pct`, `postsecondary_enrollment_pct`, `broadband_access_pct`, `stem_certified_flag`.
- **License & Terms:** Provided under Open Educational Access / Public Domain educational conventions for non-commercial learning, teaching, and research.
- **Data Dictionary:** A complete variable-by-variable description is available in [`us_education_system_data_dictionary.md`](us_education_system_data_dictionary.md).

---

## Notebook 1: Business Value of Data Visualization

**File:** [`BUSINESS VALUE OF DATA VISUALIZATION.ipynb`](BUSINESS%20VALUE%20OF%20DATA%20VISUALIZATION.ipynb)

### Purpose
This notebook approaches data visualization through the lens of executive decision-making and business strategy. Simulating a State Department of Education tasked with strategically deploying a **$50 million intervention fund** to improve statewide student math proficiency, the notebook demonstrates how visual analysis helps stakeholders audit baseline funding equity, uncover high-leverage school dynamics, and diagnose feature collinearity before building predictive models.

### Dataset Used
The notebook extracts a focused operational subset from `us_education_system_dataset.csv`:
- **Audit Subset (`allocation_data`):** `institution_name`, `level`, `control`, `title_i_status`, `per_pupil_expenditure_usd`, `percent_free_reduced_lunch`, and `math_proficiency_pct`.
- **Collinearity & Indicator Subset (`key_indicators`):** 10 continuous numeric features including spending, student-teacher ratio, meal assistance, special education, English language learner rates, math proficiency, reading proficiency, graduation rates, chronic absenteeism, and household broadband access.

### Main Steps Taken
1. **Environment Setup & Styling:** Configured Seaborn with a clean white grid theme (`seaborn-v0_8-whitegrid`) optimized for institutional reporting.
2. **Spending Distribution & Equity Baseline:** Plotted univariate per-pupil spending distributions alongside math proficiency distributions across Title I funding categories.
3. **Multi-Dimensional Lever Analysis:** Created bubble scatter plots mapping economic need against math proficiency, encoding institutional governance (`control`) and Title I status (`title_i_status`) as categorical hues and per-pupil expenditure as bubble sizes.
4. **Collinearity Audit & Correlation Heatmap:** Computed Pearson correlation coefficients ($r$) across all 10 candidate indicators and rendered a customized diverging heatmap (`blue_yellow_red`) centered at zero.
5. **Executive Synthesis:** Produced structured business takeaways addressing governance achievement gaps, Title I gradients, "Beat-the-Odds" schools, and formal modeling guidance.

### Key Visualizations & Captions

#### 1. Baseline Financial Spending & Title I Proficiency Distribution
![Financial Spending and Title 1 Distribution](images/01_business_spending_and_title1_distribution.png)
*Figure 1: Left panel shows the distribution of per-pupil spending clustering between \$12,000 and \$18,000 with a right-skewed tail reaching up to \$65,000. Right panel shows median math proficiency systematically decreasing across Title I tiers, with Title I Schoolwide institutions experiencing the lowest median scores.*

#### 2. Business Lever Analysis by Institutional Governance
![Business Lever Analysis by Governance](images/02_business_scatter_by_governance.png)
*Figure 2: Scatter plot of student economic need (% free/reduced lunch) versus math proficiency rate, colored by institutional governance (`control`) with bubble sizes reflecting per-pupil spending. Private schools cluster heavily in the low-need, high-proficiency quadrant, whereas public and charter schools span the entire socioeconomic range.*

#### 3. Business Lever Analysis by Title I Federal Status
![Business Lever Analysis by Title I Status](images/03_business_scatter_by_title1.png)
*Figure 3: Scatter plot of economic need versus math proficiency, colored by Title I status. Illustrates a clear progression from Not Applicable / Not Title I Eligible schools (low need, high proficiency) to Title I Schoolwide institutions (high need, lower proficiency).*

#### 4. Indicator Collinearity Heatmap
![Correlation Matrix of Selected Indicators](images/04_business_correlation_heatmap.png)
*Figure 4: Pairwise Pearson correlation heatmap of 10 key numeric indicators using a custom diverging colormap, highlighting strong collinear pairs such as economic need vs. broadband access ($r = -0.78$) and math vs. reading proficiency ($r = +0.77$).*

### Key Findings & Insights
- **Institutional Governance Differences:** Public schools average **46.0%** economic need and **58.0%** math proficiency, while public charters average **49.3%** need and **56.7%** proficiency. Private non-profit (**12.9%** need, **72.9%** proficiency) and private religious institutions (**13.0%** need, **70.4%** proficiency) serve substantially lower-need student populations and record higher average achievement.
- **Federal Title I Gradient:** There is a distinct tiered gradient in math performance across Title I designations:
  - *Title I Schoolwide:* **60.8%** average economic need, **51.7%** average math proficiency.
  - *Title I Targeted:* **44.0%** average economic need, **58.9%** average math proficiency.
  - *Not Title I Eligible:* **22.9%** average economic need, **67.2%** average math proficiency.
  - *Not Applicable:* **13.3%** average economic need, **72.4%** average math proficiency.
- **"Beat-the-Odds" Benchmark Schools:** Defining high-need as institutions at or above the 75th percentile of free/reduced lunch eligibility (**$\ge 59.5\%$**) and high-performance as those at or above the 75th percentile of math proficiency (**$\ge 71.8\%$**), only **50 out of 1,252 high-need schools (4.0%)** achieve "Beat-the-Odds" status. These standout schools average **76.5%** math proficiency and **\$16,762** per-pupil spending—approximately **\$1,098 (7.0%)** more per student than peer high-need schools averaging 45.7% proficiency.
- **Collinearity & Feature Selection Diagnostics:**
  - *Economic Need and Broadband Access:* Strong negative correlation ($r = -0.777$, $R^2 = 60.4\%$). Including both predictors directly in linear models introduces severe variance inflation; one should be selected or combined based on the research objective.
  - *Math and Reading Proficiency:* Strong positive alignment ($r = +0.770$, $R^2 = 59.2\%$). In a model predicting math proficiency, math is the target and reading must be treated with caution to avoid target leakage.
  - *Economic Need and Outcomes:* Moderate-to-strong inverse associations with math ($r = -0.624$) and reading ($r = -0.622$) proficiency.

### Limitations
- **Descriptive, Non-Causal Associations:** All reported differences reflect observational correlations. The \$1,098 per-pupil spending surplus in "Beat-the-Odds" schools does not prove spending caused superior academic outcomes.
- **Small Benchmark Sample Size:** The "Beat-the-Odds" group comprises only 50 institutions (4.0% of the high-need subset), meaning unobserved school-level leadership, pedagogical frameworks, or local community factors could drive these results.
- **Administrative Proxy Nature:** Title I eligibility status is an administrative funding qualification under ESEA, not an evaluation of school instructional quality or intervention efficacy.
- **Linearity Assumption:** Pearson $r$ evaluates strictly linear bivariate associations; non-linear thresholds or complex demographic interactions are not captured by the correlation matrix.

---

## Notebook 2: Introduction to Visualization for Machine Learning

**File:** [`INTRODUCTION TO VISUALIZATION FOR ML.ipynb`](INTRODUCTION%20TO%20VISUALIZATION%20FOR%20ML.ipynb)

### Purpose
This notebook introduces foundational visual techniques utilized throughout the machine learning development pipeline. It guides practitioners through inspecting raw dataset schemas, exploring feature distributions and bivariate relationships, evaluating a supervised classification model using diagnostic confusion matrix heatmaps, and projecting high-dimensional features into a 2D space using Principal Component Analysis (PCA).

### Dataset Used
The notebook utilizes `us_education_system_dataset.csv`:
- **Initial Exploration Subset (`us_edu_keymetrics_data`):** `institution_name`, `title_i_status`, `level`, `control`, `per_pupil_expenditure_usd`, `percent_free_reduced_lunch`, `math_proficiency_pct`, and `graduation_rate_pct`.
- **Model & PCA Feature Set (`numeric_cols`):** 9 clean numeric indicators: `per_pupil_expenditure_usd`, `student_teacher_ratio`, `percent_free_reduced_lunch`, `special_education_pct`, `english_learner_pct`, `math_proficiency_pct` (target), `reading_proficiency_pct`, `graduation_rate_pct`, and `chronic_absenteeism_pct`.

### Main Steps Taken
1. **Data Inspection & Exploration:** Loaded the dataset with pandas and inspected column types and non-null counts (`df.info()`).
2. **Exploratory Visualizations:** Constructed dual-panel figures comparing spending histograms (with KDE curves) to Title I math score distributions, followed by scatter plots of meal assistance versus math proficiency across governance categories.
3. **Correlation Heatmap:** Selected 9 numeric indicators, computed the full correlation matrix, and plotted an annotated heatmap with Seaborn (`cmap='Reds'`).
4. **Classification Modeling & Confusion Matrix Heatmap:**
   - Established a binary classification target: whether a school achieves math proficiency at or above the median (Class 1) or below median (Class 0).
   - Removed `math_proficiency_pct` from feature matrix $X$ (leaving 8 predictors).
   - Partitioned the data into 70% training and 30% testing subsets (`test_size=0.3`, `random_state=42`).
   - Trained a `RandomForestClassifier` with 100 decision trees.
   - Evaluated predictions via a formatted confusion matrix heatmap (`cmap='Blues'`).
5. **Dimensionality Reduction via PCA:**
   - Standardized features using `StandardScaler`.
   - Applied `PCA(n_components=2)` to compress the 8 predictors into two orthogonal dimensions.
   - Generated a 2D scatter plot colored by math performance tier, annotating axes with explained variance percentages.

### Key Visualizations & Captions

#### 1. Per-Pupil Expenditure & Title I Math Distributions
![Distribution of Per Pupil Expenditure and Boxplots](images/05_intro_ml_spending_distribution_and_boxplots.png)
*Figure 5: Side-by-side exploratory plots. The left histogram shows per-pupil spending centered around \$15,000, while the right box plot demonstrates that Not Applicable and Not Title I Eligible schools maintain higher median math proficiency compared to Title I Schoolwide and Targeted schools.*

#### 2. Socioeconomic Need vs. Math Proficiency Scatter Plot
![Socioeconomic Scatter Plot](images/06_intro_ml_socioeconomic_scatter.png)
*Figure 6: Scatter plot illustrating the negative association between student economic need (% free/reduced lunch) and school math proficiency rate, with points colored by school control type and sized by expenditure.*

#### 3. Educational & Socioeconomic Correlation Heatmap
![Correlation Matrix of Educational & Socioeconomics Indicators](images/07_intro_ml_correlation_matrix.png)
*Figure 7: Red sequential correlation matrix displaying pairwise correlations among 9 educational variables, identifying key drivers of academic performance and attendance.*

#### 4. Random Forest Confusion Matrix Diagnostic Heatmap
![Confusion Matrix Heatmap](images/08_intro_ml_confusion_matrix.png)
*Figure 8: Confusion matrix heatmap displaying classification results on 1,500 held-out test schools. The model achieved 595 True Negatives and 590 True Positives, with 136 False Positives and 179 False Negatives (79.0% overall accuracy).*

#### 5. 2D PCA Feature Projection
![2D PCA Projection](images/09_intro_ml_pca_2d_projection.png)
*Figure 9: Two-dimensional Principal Component Analysis (PCA) projection of standardized school resource and outcome metrics. Principal Component 1 captures 31.9% of total variance and separates higher-performing schools (green) from lower-performing schools (orange).*

### Key Findings & Insights
- **Bivariate Correlations:** Strongest negative correlation with math proficiency is student economic need ($r = -0.62$), while reading proficiency exhibits the highest positive correlation ($r = 0.77$). Chronic absenteeism shows a moderate negative relationship with math proficiency ($r = -0.44$) and high school graduation rates ($r = -0.42$).
- **Random Forest Model Accuracy:** The baseline classifier achieved **79.0% overall accuracy** on the 1,500-instance test set:
  - *True Negatives (Below Median correctly identified):* 595 schools.
  - *True Positives (At/Above Median correctly identified):* 590 schools.
  - *False Positives (Type I Error):* 136 schools.
  - *False Negatives (Type II Error):* 179 schools.
  - *Balanced Performance:* Precision is 0.77 for Class 0 and 0.81 for Class 1; Recall is 0.81 for Class 0 and 0.77 for Class 1 (Macro F1-score: 0.79).
- **PCA Variance Decomposition:**
  - *Principal Component 1 (PC1):* Accounts for **31.9%** of total dataset variance, aligning strongly with the socioeconomic-achievement spectrum.
  - *Principal Component 2 (PC2):* Accounts for **12.8%** of total variance.
  - *Cumulative Variance:* The two principal components combine to capture **44.7%** of the variance across all 8 input features, revealing distinct clustering between performance tiers with an overlapping boundary zone.

### Limitations
- **Information Loss via Target Binarization:** Converting continuous math proficiency into a binary indicator around the median simplifies evaluation but discards granular variance between borderline and extreme performers.
- **Row Deletion via `dropna()`:** Missing values were handled by listwise deletion, removing records with missing values rather than applying robust imputation methods.
- **Feature Exclusion:** Important categorical context (such as geographic locale, state, and school level) and secondary-specific indicators (SAT/ACT scores, AP/IB participation) were excluded to focus strictly on continuous metrics.
- **Potential Feature Dominance:** Reading proficiency ($r = 0.77$ with math) was retained as a predictor, which may dominate tree splits and mask the predictive contribution of structural variables (such as student-teacher ratios or expenditures).
- **Unexplained Variance in PCA:** The 2D PCA projection captures 44.7% of total variance, leaving 55.3% of the multidimensional feature variance unrepresented in the visualization.
- **Absence of Hyperparameter Optimization:** The Random Forest classifier was fitted using default hyperparameters (`n_estimators=100`) on a single train-test split without k-fold cross-validation.

---

## Tools & Libraries Used

| Technology | Role & Purpose |
|---|---|
| **Python 3.10+** | Core programming environment |
| **Jupyter Notebook** | Interactive computational workflow and visualization environment |
| **Pandas** | Tabular data manipulation, subset extraction, and schema inspection |
| **NumPy** | Numerical data structures and array transformations |
| **Matplotlib** | Low-level plotting framework, multi-panel layouts, and figure customization |
| **Seaborn** | High-level statistical data visualization (histograms, KDE, boxplots, scatter plots, heatmaps) |
| **Scikit-Learn** | Predictive modeling (`RandomForestClassifier`), evaluation (`confusion_matrix`), preprocessing (`StandardScaler`), and dimensionality reduction (`PCA`) |

---

## How to Run the Notebooks

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10 or higher installed. Clone or navigate to the repository directory and establish an isolated virtual environment:

```bash
# Navigate to the project root
cd "3rd SEMESTER ALTSCHOOL- MACHINE LEARNING WORKFLOW"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate
```

### 2. Install Required Dependencies
Install all required libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
Start JupyterLab or classic Jupyter Notebook from your terminal:

```bash
jupyter notebook
```
*(Alternatively, you can open the project folder in Visual Studio Code with the Jupyter extension installed.)*

### 4. Execute the Notebooks
Open either notebook and execute the cells sequentially from top to bottom:
1. First run [`BUSINESS VALUE OF DATA VISUALIZATION.ipynb`](BUSINESS%20VALUE%20OF%20DATA%20VISUALIZATION.ipynb) to inspect the executive spending audit, business lever analysis, and multicollinearity assessment.
2. Next run [`INTRODUCTION TO VISUALIZATION FOR ML.ipynb`](INTRODUCTION%20TO%20VISUALIZATION%20FOR%20ML.ipynb) to walk through the machine learning pipeline, confusion matrix heatmap, and PCA projection.

---

## Repository Structure

```text
3rd SEMESTER ALTSCHOOL- MACHINE LEARNING WORKFLOW/
│
├── BUSINESS VALUE OF DATA VISUALIZATION.ipynb   # Notebook 1: Strategic visual audit & feature collinearity
├── INTRODUCTION TO VISUALIZATION FOR ML.ipynb   # Notebook 2: ML visualization, Random Forest, & PCA
├── us_education_system_dataset.csv             # Primary synthetic dataset (5,000 records × 35 variables)
├── us_education_system_data_dictionary.md       # Comprehensive variable metadata and NCES definitions
├── requirements.txt                            # Python environment dependencies
├── README.md                                   # Comprehensive project documentation
│
└── images/                                     # High-resolution chart visual outputs
    ├── 01_business_spending_and_title1_distribution.png
    ├── 02_business_scatter_by_governance.png
    ├── 03_business_scatter_by_title1.png
    ├── 04_business_correlation_heatmap.png
    ├── 05_intro_ml_spending_distribution_and_boxplots.png
    ├── 06_intro_ml_socioeconomic_scatter.png
    ├── 07_intro_ml_correlation_matrix.png
    ├── 08_intro_ml_confusion_matrix.png
    └── 09_intro_ml_pca_2d_projection.png
```

---

## Author

**Eberechi Bessie Donalds**  
Data Science Student, AltSchool Africa  
*3rd Semester — Machine Learning Workflow Track*
