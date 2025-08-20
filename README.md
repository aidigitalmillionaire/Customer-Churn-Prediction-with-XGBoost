# Customer-Churn-Prediction-with-XGBoost
🔄 Customer Churn Prediction with XGBoost – Built a machine learning model to identify customers at risk of leaving, leveraging demographic, behavioral, and transaction data. The project includes data preprocessing, feature engineering, EDA, and model optimization with XGBoost.

# 🔄 Customer Churn Prediction with XGBoost  

This project uses **XGBoost** to predict customer churn based on demographic, usage, and behavioral data.  
The goal is to help businesses identify at-risk customers early and design effective retention strategies.  

---

## 🚀 Project Overview
- Perform **EDA & feature engineering** on customer data.
- Train and tune an **XGBoost model** for churn classification.
- Evaluate performance with metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- Use **feature importance analysis** to identify main churn drivers.
- Provide insights to support **business decision-making**.

---

## 📂 Repository Structure

├── README.md # Project overview

├── customer_churn_xgboost.ipynb # Main Colab notebook

├── requirements.txt # Python dependencies

├── docs/

│ └── DESCRIPTION.md # Detailed description

├── data/

│ └── dataset_link.txt # Dataset source

├── images/

│ └── feature_importance.png # Visualization of feature importance


---

## 📊 Dataset  
Dataset used: **Telco Customer Churn**  
🔗 [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/cavinlobo/cleaned-dataset-for-telco-customer-churn/data)  

---

## ⚙️ Installation  

Clone repo:  

git clone https://github.com/your-username/Customer-Churn-Prediction-with-XGBoost.git
cd Customer-Churn-Prediction-with-XGBoost

## 📊 Dataset  
Dataset used: **Telco Customer Churn**  
🔗 [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/cavinlobo/cleaned-dataset-for-telco-customer-churn/data)  

---

# Install dependencies:

pip install -r requirements.txt

# ▶️ Usage

Open customer_churn_xgboost.ipynb in Google Colab.

Mount dataset or load directly from Kaggle.

Run all cells to preprocess data, train the model, and generate results.

# 📈 Results

XGBoost achieved high accuracy and ROC-AUC score compared to baseline models.

Feature importance showed that tenure, contract type, and monthly charges were the strongest churn predictors.

Business can reduce churn by offering retention plans for customers with short tenure or high monthly charges.


---

# **6. Add Description File**  

Create `docs/DESCRIPTION.md`:  

```markdown
# Project Description: Customer Churn Prediction with XGBoost  

Customer churn is a critical business problem that impacts long-term revenue. This project demonstrates how to use **XGBoost**, a powerful gradient boosting algorithm, to predict churn and understand the factors that drive it.  

#Key Steps:  
- Data preprocessing and handling missing values  
- Exploratory Data Analysis (EDA) with visualizations  
- Feature engineering and encoding categorical variables  
- Model training using XGBoost  
- Hyperparameter tuning for improved performance  
- Evaluation using ROC-AUC, F1-score, confusion matrix  
- Feature importance analysis for business insights   

# Business Impact:  
By predicting churn and identifying its drivers, businesses can take **proactive retention measures**, lower churn rates, and increase **customer lifetime value (CLV)**.



