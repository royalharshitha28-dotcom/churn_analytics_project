import pandas as pd
import joblib

rf_model = joblib.load('churn_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

input_data = {
    'Gender': [1],  
    'SeniorCitizen': [0],
    'Tenure': [12],
    'PaperlessBilling': [1],
    'MonthlyCharges': [75.0],
    'TotalCharges': [900.0],
    'Contract_One year': [1],
    'Contract_Two year': [0],
    'PaymentMethod_Credit card (automatic)': [1],
    'PaymentMethod_Electronic check': [0],
    'PaymentMethod_Mailed check': [0]
}

input_df = pd.DataFrame(input_data)

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

features_to_scale = ['Tenure', 'MonthlyCharges', 'TotalCharges']
input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])

prediction = rf_model.predict(input_df)[0]
probability = rf_model.predict_proba(input_df)[0][1]

print("Prediction:", prediction)
print("Probability:", probability)
