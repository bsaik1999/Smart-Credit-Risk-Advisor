import pandas as pd
import numpy as np

# Phase 1: Load dataset
file_path = r"C:\Users\SaiKrishna\OneDrive\Desktop\UCI_Credit_Card.csv"
df = pd.read_csv(file_path)

print("Initial data shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())

# Phase 2: Feature Engineering
df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)  # 4 = others
df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)  # 3 = others

bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
df['avg_pay_amt'] = df[pay_cols].mean(axis=1)

def slope_of_series(series):
    x = np.arange(len(series))
    y = series.values
    if np.all(y == 0):
        return 0
    return np.polyfit(x, y, 1)[0]

df['bill_amt_trend'] = df[bill_cols].apply(slope_of_series, axis=1)
df['pay_amt_trend'] = df[pay_cols].apply(slope_of_series, axis=1)
df['credit_util_ratio'] = df['avg_bill_amt'] / df['LIMIT_BAL']
df['payment_ratio'] = df['avg_pay_amt'] / df['avg_bill_amt'].replace(0, np.nan)
df['payment_ratio'] = df['payment_ratio'].fillna(0)

pay_status_cols = [f'PAY_{i}' for i in ['0','2','3','4','5','6']]
df['mean_pay_status'] = df[pay_status_cols].mean(axis=1)
df['max_pay_status'] = df[pay_status_cols].max(axis=1)

print("Sample engineered features:\n", df[['avg_bill_amt', 'avg_pay_amt', 'bill_amt_trend', 'pay_amt_trend',
                                           'credit_util_ratio', 'payment_ratio', 'mean_pay_status',
                                           'max_pay_status']].head())

# Save processed CSV to your specific path
save_path = r"C:\Users\SaiKrishna\OneDrive\Desktop\archive (3)\credit_data_processed.csv"
df.to_csv(save_path, index=False)
print(f"✅ Processed CSV saved to: {save_path}")
