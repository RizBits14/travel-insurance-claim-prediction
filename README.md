# travel-insurance-claim-prediction
Travel Insurance Claim Prediction & Customer Segmentation

This project applies machine learning techniques to predict whether a customer will make a travel insurance claim and to segment customers using unsupervised clustering.
It demonstrates a full workflow from data exploration to model evaluation and includes a final report with visual insights.

**Dataset**

- **Total Records** -> 63,326
- **Target Variables** -> `Claim` (1 = Claim, 0 = No Claim)
- **Class Imbalace** -> Only ~ 1.5% claim

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

**1. Data Cleaning and Prepocessing**
- Handled missing values (Gender → "Unknown")
- Encoded categorical variables
- Scaled numerical features
- Addressed imbalance using class_weight='balanced'

**2. Explanatory Data Analysis**
- Claim distribution
- Correlation heatmap
- Claim rate by agency, product and region

**3. Model Building (Supervised Learning)
- Logistic Regression
- Desision Tree
- Neural Network (MLP)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, AUC

**4. Unsupervised Clustering (K-Means)
- Grouped customers into three clusters
- Evaluated using Silhouette Score (0.28)
