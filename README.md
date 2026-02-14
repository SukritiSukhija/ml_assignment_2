# Adult Income Classification – Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related features.

The task is a binary classification problem.

---

## 2. Dataset Description

Dataset Used: Adult Income Dataset (UCI Repository)

Number of Instances: ~48,000  
Number of Features: 14 original features  
Target Variable: Income (<=50K or >50K)

Features include:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

Preprocessing Steps:
- Removed missing values
- Encoded categorical variables using one-hot encoding
- Scaled numerical features using StandardScaler
- Split dataset into training and testing sets (80-20 split)

---

## 3. Models Used and Evaluation Metrics

The following models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics Used:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-----------|----------|------|------------|--------|------|------|
| Logistic Regression | 0.81 | 0.90 | 0.57 | 0.85 | 0.68 | 0.58 |
| Decision Tree | 0.85 | 0.90 | 0.72 | 0.66 | 0.69 | 0.60 |
| KNN | 0.82 | 0.84 | 0.65 | 0.58 | 0.61 | 0.50 |
| Naive Bayes | 0.42 | 0.68 | 0.29 | 0.97 | 0.44 | 0.23 |
| Random Forest | 0.86 | 0.92 | 0.79 | 0.58 | 0.67 | 0.60 |
| XGBoost | 0.87 | 0.93 | 0.77 | 0.66 | 0.71 | 0.63 |

---

## 5. Observations

- XGBoost achieved the best overall performance across most metrics.
- Random Forest also performed strongly and provided balanced precision and recall.
- Logistic Regression performed well but had lower capability to capture complex patterns.
- Decision Tree improved after tuning but is prone to overfitting.
- KNN showed moderate performance.
- Naive Bayes achieved very high recall but very low precision, indicating many false positives.

Overall, ensemble methods (Random Forest and XGBoost) outperformed individual models due to their ability to capture non-linear relationships.

---

## 6. Streamlit Application

The Streamlit application includes:
- Model selection dropdown
- Dataset upload option
- Display of predictions
- Display of evaluation metrics
- Confusion matrix visualization

---

## 7. Repository Structure
ml_assignment_2/
│-- app.py
│-- utils.py
│-- models.py
│-- requirements.txt
│-- saved_models/
│-- notebooks/

## 8. How to Run the Project Locally

1. Clone the repository:
   git clone <your-repo-link>

2. Navigate to project folder:
   cd ml_assignment_2

3. Install dependencies:
   pip install -r requirements.txt

4. Run Streamlit app:
   streamlit run app.py



## 9. Live Streamlit App

Deployed App Link:
https://your-app-link.streamlit.app