import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
import os

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("viridis")

def perform_eda_and_cleaning():
    print("Loading data...")
    df = pd.read_csv('customer_churn.csv')
    
    os.makedirs('images', exist_ok=True)
    
    # 1. EDA before cleaning
    print("Generating EDA visualizations...")
    
    # Churn Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Customer Churn Distribution')
    plt.tight_layout()
    plt.savefig('images/churn_distribution.png')
    plt.close()
    
    # Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Tenure', hue='Churn', multiple='stack', bins=30)
    plt.title('Tenure Distribution by Churn')
    plt.tight_layout()
    plt.savefig('images/tenure_vs_churn.png')
    plt.close()
    
    # Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
    plt.title('Monthly Charges by Churn Status')
    plt.tight_layout()
    plt.savefig('images/monthly_charges_vs_churn.png')
    plt.close()
    
    # 2. Data Cleaning
    print("Cleaning data...")
    # Fill missing TotalCharges with median
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Save a clean copy for the dashboard
    df.to_csv('customer_churn_clean.csv', index=False)
    
    # 3. Feature Engineering & Preprocessing for Modeling
    print("Preprocessing for modeling...")
    df_model = df.drop('CustomerID', axis=1) # Drop ID column
    
    # Label encoding for binary categorical
    le = LabelEncoder()
    df_model['Gender'] = le.fit_transform(df_model['Gender'])
    df_model['PaperlessBilling'] = le.fit_transform(df_model['PaperlessBilling'])
    df_model['Churn'] = le.fit_transform(df_model['Churn'])
    
    # One-Hot Encoding for multi-class categorical
    df_model = pd.get_dummies(df_model, columns=['Contract', 'PaymentMethod'], drop_first=True)
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_model.corr(), annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png')
    plt.close()
    
    return df_model, df

def train_model(df_model):
    print("Training Random Forest model...")
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features (Tenure, MonthlyCharges, TotalCharges)
    scaler = StandardScaler()
    features_to_scale = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    print("\nModel Evaluation Metrics:")
    print("-" * 30)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    
    # Feature Importance Visualization
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[X.columns[i] for i in indices])
    plt.title('Feature Importances in Random Forest')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.close()
    
    # Save the model and scaler
    joblib.dump(rf_model, 'churn_rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Also save column names for prediction via app
    joblib.dump(list(X.columns), 'model_columns.pkl')
    
    print("Model and artifacts saved successfully.")

if __name__ == "__main__":
    df_model, original_df = perform_eda_and_cleaning()
    train_model(df_model)
    print("Analysis script completed.")
