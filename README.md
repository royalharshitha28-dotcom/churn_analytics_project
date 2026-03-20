# End-to-End Customer Churn Analytics Project

## Problem Statement
Customer retention is one of the most pressing issues for subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones. The objective of this project is to analyze customer data for a streaming service, identify the key drivers of customer churn, and build a predictive machine learning model to flag high-risk customers before they cancel their subscriptions. This enables the business to take proactive measures, such as offering targeted discounts or improved customer service.

## Project Structure
- `data_generator.py`: Generates the realistic synthetic dataset (`customer_churn.csv`).
- `analysis.py`: Performs Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, and Model Training (Random Forest).
- `app.py`: Streamlit dashboard for real-time data exploration and churn prediction.
- `requirements.txt`: Python package dependencies.
- `images/`: Directory containing generated EDA plots.
- `*.pkl`: Saved model and preprocessing artifacts.

## Setup & Deployment Instructions

### 1. Environment Setup
It is recommended to use a virtual environment.
```bash
python -m venv venv
```
Activate the virtual environment:
- **Windows:** `.\venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Model
Run the data generator to create `customer_churn.csv`:
```bash
python data_generator.py
```
Run the analysis script to generate EDA plots, clean the data, and train the Random Forest model:
```bash
python analysis.py
```

### 4. Run the Streamlit Dashboard
Launch the interactive dashboard:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your web browser.

## Business Insights & Recommendations
1. **Focus on Early Retention:** The data shows that customers with a tenure of less than 12 months are at the highest risk of churning. **Recommendation:** Implement a robust onboarding program and offer loyalty incentives during the first year of service.
2. **Review Pricing Strategy:** High monthly charges correlate strongly with increased churn probability. **Recommendation:** Introduce flexible, tiered pricing plans or "lite" subscriptions for budget-conscious users.
3. **Incentivize Long-Term Contracts:** Customers on month-to-month contracts churn at a substantially higher rate than those on 1-year or 2-year contracts. **Recommendation:** Offer discounted rates for users who commit to annual billing cycles to lock in revenue and reduce month-to-month volatility.
