# Telco Customer Churn Prediction
This project predicts customer churn for a telecom company using a **Logistic Regression** model. Users can explore the dataset, visualize class imbalance, train a model, evaluate it, and make predictions for new customers.

---

## **Features**

1. **Dataset Exploration**
   - View column names and data types.
   - Check unique values in each column.

2. **Class Imbalance Visualization**
   - Countplot of churned vs active customers.

3. **Model Training**
   - Preprocessing: Label encoding, one-hot encoding, and scaling.
   - Logistic Regression model with balanced class weights.
   - Split dataset into training and testing sets.

4. **Customer Prediction**
   - Input customer details and predict churn (0 = active, 1 = churned).

5. **Model Evaluation**
   - Confusion matrix visualization.
   - Accuracy, Precision, Recall, and F1 Score metrics.
   - Comparative bar plot of evaluation metrics.

---

## **Installation**

1. Clone this repository or download the code files.
2. Install required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

