# travel-insurance-claim-prediction
Travel Insurance Claim Prediction & Customer Segmentation

This project applies machine learning techniques to predict whether a customer will make a travel insurance claim and to segment customers using unsupervised clustering.
It demonstrates a full workflow from data exploration to model evaluation and includes a final report with visual insights.

**Dataset**

- **Total Records** -> 63,326
- **Target Variable** -> `Claim` (1 = Claim, 0 = No Claim)
- **Class Imbalance** -> Only ~ 1.5% claim

## Dataset Features

| Feature          | Description                                |
|------------------|--------------------------------------------|
| **Agency**       | Name of the travel agency                  |
| **Agency Type**  | Type of agency (e.g., Corporate)           |
| **Product Name** | Name of the insurance product              |
| **Duration**     | Travel duration in days                    |
| **Destination**  | Destination country                        |
| **Net Sales**    | Total sales amount                         |
| **Age**          | Age of the customer                        |
| **Gender**       | Gender of the customer                     |
| **Claim**        | Whether a claim was made (`Yes` or `No`)   |

**Workflow**

**1. Data Cleaning and Preprocessing**
- Handled missing values (Gender → "Unknown")
- Encoded categorical variables
- Scaled numerical features
- Addressed imbalance using class_weight='balanced'

**2. Exploratory Data Analysis**
- Claim distribution
- Correlation heatmap
- Claim rate by agency, product and region

**3. Model Building (Supervised Learning)**
- Logistic Regression
- Decision Tree
- Neural Network (MLP)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, AUC

**4. Unsupervised Clustering (K-Means)**
- Grouped customers into three clusters
- Evaluated using Silhouette Score (0.28)

## Model Performance (Supervised)

| **Metric**   | **Logistic Regression**  | **Decision Tree** | **Neural Network** |
|--------------|--------------------------|-------------------|--------------------|
| **Accuracy** | 0.7952                   | 0.9713            | 0.9854             |
| **Precision**| 0.0517                   | 0.0680            | 0.0000             |
| **Recall**   | 0.7514                   | 0.0757            | 0.0000             |
| **AUC**      | 0.8312                   | 0.5319            | 0.8383             |

**Key Insight**
- Logistic Regression performed best for identifying claims due to high recall and balanced AUC.

**Future Enhancements**
- Apply **SMOTE** to improve class balance
- Use advanced models like **XGBoost** or **LightGBM**
- Deploy as a **web app** using **Streamlit** or **Flask**
