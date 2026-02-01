# Telco Customer Churn Prediction

## Problem Statement
The objective of this project is to predict customer churn for a telecommunications company using machine learning. By identifying customers at risk of leaving, the company can proactively implement retention strategies. The solution involves data preprocessing, training six different classification models, evaluating their performance, and deploying a user-friendly Streamlit application for real-time predictions.

## Dataset Description

*   **Name**: Telco Customer Churn
*   **Number of rows**: 7043
*   **Number of features**: 19 (Independent variables)
*   **Target variable**: `Churn` (Yes/No)

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.81 | 0.86 | 0.68 | 0.57 | 0.62 | 0.50 |
| Decision Tree | 0.74 | 0.66 | 0.51 | 0.49 | 0.50 | 0.32 |
| KNN | 0.76 | 0.78 | 0.56 | 0.49 | 0.53 | 0.37 |
| Naive Bayes | 0.76 | 0.83 | 0.54 | 0.76 | 0.63 | 0.47 |
| Random Forest | 0.80 | 0.84 | 0.68 | 0.49 | 0.57 | 0.46 |
| XGBoost | 0.80 | 0.83 | 0.66 | 0.52 | 0.58 | 0.45 |

## Observations Table

| Model | Observations |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest accuracy (81%) and a balanced F1 score. Effective linear separation for this dataset. |
| **Decision Tree** | Lower performance metrics across the board (Accuracy 74%), indicating potential overfitting or inability to capture complex patterns without ensemble methods. |
| **KNN** | Moderate performance (Accuracy 76%). Sensitivity to the scale of features was handled by MinMax scaling, but it lagged behind tree-based ensembles. |
| **Naive Bayes** | Highest Recall (76%), making it excellent for identifying potential churners, though at the cost of lower Precision (54%). |
| **Random Forest** | Strong performance (Accuracy 80%) comparable to Logistic Regression but with slightly lower Recall. Good generalization. |
| **XGBoost** | Competitive performance (Accuracy 80%) with robust metrics. Slightly lower recall compared to Naive Bayes but better precision. |

## Repository Structure

```
├── app.py
├── requirements.txt
├── README.md
├── ML_Assignment_2.ipynb
├── Telco-Customer-Churn.csv
└── model/
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── confusion_matrices.pkl
```

## How to Run Locally

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## How to Deploy on Streamlit Cloud

1.  Push this repository to GitHub.
2.  Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3.  Click "New app" and select this repository.
4.  Set the **Main file path** to `app.py`.
5.  Click **Deploy**.
