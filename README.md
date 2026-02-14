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

## 5. Model-wise Observations

### Logistic Regression
Performed well as a baseline linear classifier with strong AUC. However, due to its linear decision boundary, it was limited in capturing complex non-linear feature interactions.

### Decision Tree
After tuning, performance improved significantly. It captures non-linear relationships effectively but is prone to overfitting compared to ensemble methods.

### K-Nearest Neighbors (KNN)
Showed moderate performance. Its effectiveness is influenced by feature scaling and high dimensionality caused by one-hot encoding.

### Naive Bayes
Achieved very high recall but extremely low precision, indicating a large number of false positives. The independence assumption of features does not hold well for this dataset.

### Random Forest
Delivered strong and stable performance with balanced precision and recall. Ensemble averaging helped reduce overfitting seen in a single decision tree.

### XGBoost
Achieved the best overall performance across most metrics including AUC, F1-score, and MCC. Its boosting mechanism effectively captured complex feature interactions.

Overall, ensemble models (Random Forest and XGBoost) outperformed individual classifiers due to their ability to model non-linear relationships and reduce variance.
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
│-- data/


## 8. How to Run the Project Locally

1. Clone the repository:
   git clone https://github.com/SukritiSukhija/ml_assignment_2.git

2. Navigate to project folder:
   cd ml_assignment_2

3. Install dependencies:
   pip install -r requirements.txt

4. Run Streamlit app:
   streamlit run app.py



## 9. Live Streamlit App

Deployed App Link:
https://mlassignment2-gre6jxyroevqdam4qubx6a.streamlit.app/
