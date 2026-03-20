import pandas as pd
import numpy as np
import os

def generate_customer_data(num_samples=5000):
    """Generates a realistic synthetic customer churn dataset."""
    
    np.random.seed(42)
    
    # Customer IDs
    customer_ids = [f'CUST_{str(i).zfill(5)}' for i in range(1, num_samples + 1)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], num_samples)
    senior_citizen = np.random.choice([0, 1], num_samples, p=[0.85, 0.15])
    
    # Account Information
    tenure_months = np.random.randint(1, 73, num_samples)
    
    # Service Information
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract = np.random.choice(contract_types, num_samples, p=[0.5, 0.25, 0.25])
    
    paperless_billing = np.random.choice(['Yes', 'No'], num_samples, p=[0.6, 0.4])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples)
    
    # Financials
    # Base charges vary based on contract
    monthly_charges = []
    for c in contract:
        if c == 'Month-to-month':
            charge = np.random.uniform(50, 110)
        elif c == 'One year':
            charge = np.random.uniform(40, 90)
        else:
            charge = np.random.uniform(30, 80)
        monthly_charges.append(charge)
        
    monthly_charges = np.round(monthly_charges, 2)
    
    # Total charges (introducing some random NaN values to simulate real-world data)
    total_charges = np.round(monthly_charges * tenure_months, 2)
    
    # Inject missing values in TotalCharges
    num_missing = int(num_samples * 0.01) # 1% missing
    missing_indices = np.random.choice(num_samples, num_missing, replace=False)
    total_charges = total_charges.astype(object) # Convert to object to hold None
    for idx in missing_indices:
        total_charges[idx] = np.nan
        
    # Inject outliers in MonthlyCharges
    num_outliers = int(num_samples * 0.005) # 0.5% outliers
    outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
    for idx in outlier_indices:
        monthly_charges[idx] = np.random.uniform(200, 300)

    # Churn Logic (Creating a realistic distribution based on features)
    churn_prob = np.zeros(num_samples)
    
    # Base probability
    churn_prob += 0.2
    
    # Month-to-month contracts have higher churn
    churn_prob[contract == 'Month-to-month'] += 0.15
    churn_prob[contract == 'Two year'] -= 0.1
    
    # Shorter tenure leads to higher churn
    churn_prob[tenure_months < 12] += 0.1
    churn_prob[tenure_months > 48] -= 0.1
    
    # High monthly charges increase churn risk
    churn_prob[monthly_charges > 85] += 0.05
    
    # Electronic check users often have higher churn
    churn_prob[payment_method == 'Electronic check'] += 0.05
    
    # Ensure probabilities are bounded between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Generate actual churn labels
    churn = np.array(['Yes' if np.random.random() < p else 'No' for p in churn_prob])
    
    # Create DataFrame
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': gender,
        'SeniorCitizen': senior_citizen,
        'Tenure': tenure_months,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_customer_data(num_samples=5000)
    
    # Create project directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'customer_churn.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset generated and saved to: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nSample Data:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())
