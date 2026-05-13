# 🏦 Indian Bank Loan Underwriting System

An end-to-end Machine Learning pipeline and Streamlit Web Application designed to predict loan approvals based on Indian banking standards. The system evaluates applicant profiles, calculates critical financial metrics (FOIR, LTV), and provides a robust, confidence-backed decision using an ensemble of Machine Learning models.

### Live Dashboard
![Input Form](outputs/ui_screenshot_1.png)
![Prediction Results](outputs/ui_screenshot_2.png)

## 📌 Project Overview
Traditional loan underwriting processes are heavily manual, prone to human error, and time-consuming. This project automates the risk assessment process by training a **Random Forest Classifier** on over 4,000 real-world banking records. 

The application evaluates:
- **Demographics**: Dependents, Education, Employment Type.
- **Financial Capacity**: Annual Gross Income, Liquid Assets.
- **Credit Bureau**: CIBIL Scores (300-900).
- **Immovable Assets**: Residential, Commercial, and Luxury property valuations.

## ✨ Key Features
- **Accurate Predictions**: Achieves **99.5% accuracy** using Random Forest.
- **Indian Banking Context**: Uses Lakhs/Crores for inputs, strictly enforces CIBIL bounds, and natively calculates Indian home loan term maximums.
- **Key Risk Indicators (KRI)**: Automatically estimates Debt-to-Service Ratios (FOIR/EMI) and Loan-to-Value (LTV) percentages on the fly.
- **Interactive UI**: A beautiful, corporate-styled dashboard built entirely with Streamlit and custom CSS.
- **Comprehensive EDA**: Generates publication-ready visualizations of the underlying data distributions.

---

## 🛠️ Technology Stack
- **Language**: Python 3.x
- **Machine Learning**: Scikit-Learn (`RandomForest`, `SVM`, `DecisionTree`, `LogisticRegression`, `NaiveBayes`)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Web Interface**: Streamlit

---

## 📊 Exploratory Data Analysis (EDA)
The system includes an automated EDA module that generates insights on how different factors influence loan approvals.

### CIBIL Score vs Approvals
CIBIL score is the strongest predictor of loan approval. As the distribution shows, applicants with scores > 700 have a near-guaranteed approval rate, while scores < 600 face immediate rejection.
![CIBIL Analysis](outputs/02_cibil_vs_approval.png)

### Asset Valuation Impact
Higher residential and luxury assets correlate strongly with loan approvals, providing banks with sufficient collateral.
![Asset Analysis](outputs/04_assets_analysis.png)

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
pip install streamlit scikit-learn pandas numpy matplotlib seaborn kagglehub
```

### 1. Download Dataset
The system is built to ingest the Kaggle dataset seamlessly. Run the fetch script to download the latest data:
```bash
python fetch_kaggle_data.py
```

### 2. Run the Machine Learning Pipeline
To train the models, engineer features, and generate the EDA plots, run the main orchestrator:
```bash
python main.py
```
This will automatically evaluate 5 different algorithms and select the best performing one.

### 3. Launch the Web Application
To interact with the AI through the industrial UI dashboard:
```bash
python -m streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 🏗️ Project Structure
```text
project/
│
├── data/                  # Contains the raw CSV datasets
├── outputs/               # Saved EDA plots and model summaries
├── src/
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── eda.py                 # Visualization logic
│   ├── model_training.py      # ML algorithm training & evaluation
│   └── predict.py             # Inference module
│
├── app.py                 # Streamlit web dashboard
├── main.py                # Pipeline orchestrator
├── fetch_kaggle_data.py   # Automated dataset downloader
└── README.md              # Project documentation
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Disclaimer: This is a predictive tool intended for educational purposes and should not be used as the sole deciding factor for actual financial underwriting without human oversight.*
