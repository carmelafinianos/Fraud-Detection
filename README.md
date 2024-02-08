# Fraud-Detection
# Fraud Detection in Financial Transactions

## Overview
This project aims to identify fraudulent transactions in financial datasets using Python's popular data analysis and machine learning libraries. By employing a combination of data preprocessing, feature engineering, and machine learning models, we strive to accurately classify transactions as fraudulent or legitimate.

## Dependencies
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

## Dataset
The dataset, `synth_training_fe.csv`, contains various features related to financial transactions, including transaction dates, amounts, and customer information.

## Features
- **Preprocessing**: Converts dates to datetime objects, extracts date and time, calculates age from date of birth, and handles missing values.
- **Feature Engineering**: Adds new features such as the total number of transactions per card, maximum transaction amount per day, and daily transaction counts.
- **Label Encoding**: Encodes non-numeric categorical variables for model compatibility.
- **Machine Learning Models**: Utilizes Random Forest Classifier and Support Vector Machine (SVM) for predicting fraudulent transactions.
- **Evaluation**: Assesses model performance using accuracy, precision, recall, confusion matrix, and ROC curve.

## Usage
1. **Data Preparation**: Load the dataset and perform necessary preprocessing and feature engineering steps.
2. **Model Training**: Split the data into training and test sets, then train the models using the training set.
3. **Prediction and Evaluation**: Make predictions on the test set and evaluate the models' performances through various metrics.
4. **Insight Extraction**: Utilize confusion matrices, classification reports, and feature importance plots to gain insights into the model's behavior and important features.

## Code Snippets

### Data Loading and Preprocessing
```python
import pandas as pd
data = pd.read_csv("synth_training_fe.csv")
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
```

### Model Training and Evaluation
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred)}')
```

### Feature Importance Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = model.feature_importances_
sns.barplot(x=feature_importance, y=X.columns, palette='viridis')
```

## Results
- The project demonstrates the effectiveness of machine learning models in detecting fraudulent transactions.
- Performance metrics, such as accuracy, precision, and recall, provide insights into model strengths and areas for improvement.

## Conclusion
This project highlights the potential of machine learning in enhancing the security and reliability of financial transactions by detecting fraud. Future work could explore more sophisticated models and feature engineering techniques to further improve performance.
