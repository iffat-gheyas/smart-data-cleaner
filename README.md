# Smart Data Cleaner

**A modular Python and Streamlit application for automated data cleaning, validation, preprocessing and machine learning data preparation.**

Smart Data Cleaner provides an interactive environment for identifying and addressing data quality problems in Excel and CSV datasets. It supports general-purpose data cleaning and specialised preprocessing for both cross-sectional and time-series machine learning applications.

The application combines automated data analysis with user-guided decision-making. Rather than making unsupported assumptions about ambiguous data, it requests clarification where necessary and maintains an audit trail of the transformations performed.

## Key Features

### 1. General Data Cleaning

The application supports a wide range of data cleaning and standardisation operations, including:

- Detection and removal of duplicate records.
- Identification and treatment of missing values.
- Statistical imputation using mean, median, mode, KNN and multiple imputation methods.
- Detection and flagging of outliers using statistical methods.
- Text cleaning and categorical value standardisation.
- Date, time and duration parsing and standardisation.
- Unit and currency conversion.
- Validation and formatting of phone numbers and email addresses.
- Standardisation of country, state and geographical information.
- Splitting and restructuring columns containing combined information.
- Interactive correction of inconsistent or invalid values.

### 2. Machine Learning Data Preparation

Smart Data Cleaner includes dedicated preprocessing workflows for preparing datasets for machine learning.

**Cross-Sectional Data**

- General data cleaning and validation.
- Missing-value imputation.
- Categorical variable encoding, including one-hot, binary and ordinal encoding.
- Preparation of structured datasets for downstream machine learning algorithms.

**Time-Series Data**

- Identification and validation of time-series structures.
- Date/time parsing and temporal consistency checks.
- Frequency identification and aggregation.
- Detection and treatment of missing timestamps and observations.
- Time-series imputation.
- Plausibility checks and temporal data validation.
- Preparation of time-series datasets for forecasting and predictive modelling.

### 3. Dataset Integration

The application supports merging and combining multiple datasets through:

- Row-wise concatenation.
- Column-wise joins using different matching strategies.
- Single or composite matching keys.
- Identification and management of overlapping columns.
- Generation of merge summaries and audit logs.

### 4. Interactive Data Quality Assessment

Smart Data Cleaner combines automated preprocessing with user-guided decisions.

When data values are ambiguous, users can review potential problems, provide corrections, specify appropriate treatments or retain the original values.

This approach reduces the risk of introducing errors through unsupported assumptions.

### 5. Reproducibility and Reporting

The application produces:

- Cleaned datasets in Excel format.
- Summaries of detected data quality problems and corrective actions.
- Detailed processing logs.
- Separate records of outlier and validation flags.

These outputs support transparency, traceability and reproducibility in data preprocessing.

## Software Architecture

Smart Data Cleaner uses a modular Python architecture consisting of approximately 57 specialised processing modules.

The modules address individual data preparation tasks, while the Streamlit interface manages user interaction and workflow execution.

Key files include:

- `app.py` — Main Streamlit application.
- `01_ingestion.py` — Data ingestion.
- `11_question_generation.py` — Generation of data-specific clarification questions.
- `12_apply_answers_and_run.py` — Execution of the preprocessing workflow.
- `requirements.txt` — Python dependencies.

The numbered Python modules are maintained in the same directory as `app.py`.

## Technologies

**Programming language:** Python

**User interface:** Streamlit

**Data processing:** pandas, NumPy

**Machine learning and statistical methods:** scikit-learn and related scientific Python libraries

**Spreadsheet processing:** openpyxl, XlsxWriter

**Additional libraries:** phonenumbers, email-validator, pycountry, word2number and PyTorch

## Installation and Execution

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

The application can run locally or in a suitable hosted Python environment.

## Data Handling

Uploaded datasets are processed in memory, with downloadable outputs generated after preprocessing. The application is designed to avoid persistent storage of uploaded datasets during normal processing.

Users should avoid uploading confidential or sensitive information to hosting environments without first assessing the applicable data protection and security requirements.

## Known Limitations

- Address parsing primarily supports UK postcode-based structures.
- Ambiguous time expressions may require user clarification.
- Month-to-day and year-to-day conversions use approximate calendar averages.
- Some specialised formatting options are not implemented.

## Project Purpose

Smart Data Cleaner was developed as a practical Python application for data quality assessment, automated preprocessing and the preparation of datasets for statistical analysis and machine learning.

The project demonstrates modular Python software development, data processing, interactive application design and the integration of statistical and machine learning preprocessing techniques.
