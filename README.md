# Credit Risk Modelling

This project aims to develop a machine learning model to predict the likelihood of loan approval based on applicant information. It encompasses data preprocessing, feature engineering, model training using XGBoost, and deployment as a standalone executable using PyInstaller.

## 📁 Project Structure

```
Credit_Risk_Modelling/
├── data/
│   ├── case_study1.xlsx
│   ├── case_study2.xlsx
│   └── Final_Predictions.xlsx
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher
* Recommended: Create and activate a virtual environment

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vaibhavbunny/Credit_Risk_Modelling.git
   cd Credit_Risk_Modelling
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that `openpyxl` is installed for Excel file operations.*

3. **Run the application:**

   ```bash
   python main.py
   ```

## 🧠 Project Workflow

1. **Data Loading:**

   * Reads input datasets from `data/case_study1.xlsx` and `data/case_study2.xlsx`.

2. **Data Preprocessing:**

   * Handles missing values.
   * Encodes categorical variables using one-hot encoding.
   * Ensures all encoded features are in `uint8` format for efficiency.

3. **Feature Engineering:**

   * Calculates Variance Inflation Factor (VIF) to detect multicollinearity.
   * Performs Chi-Square tests to assess the significance of categorical features.

4. **Model Training:**

   * Utilizes `XGBoost` classifier with `multi:softmax` objective for multi-class classification.
   * Implements label encoding for target variables.

5. **Prediction and Output:**

   * Generates predictions on the test dataset.
   * Saves the final predictions to `data/Final_Predictions.xlsx`.

6. **Deployment:**

   * Converts the Python script into a standalone executable using PyInstaller:

     ```bash
     pyinstaller --onefile main.py

## 📌 Requirements

* pandas
* numpy
* xgboost
* scikit-learn
* openpyxl
* statsmodels
* pyinstaller

*All dependencies are listed in `requirements.txt`.*

