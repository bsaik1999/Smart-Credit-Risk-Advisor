import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv(r"C:\Users\SaiKrishna\OneDrive\Desktop\UCI_Credit_Card.csv")

# Prepare time series data: past 5 billing amounts and payments as features
bill_cols_5 = [f'BILL_AMT{i}' for i in range(1, 6)]
pay_cols_5 = [f'PAY_AMT{i}' for i in range(1, 6)]

X_bill = df[bill_cols_5].values
X_pay = df[pay_cols_5].values

# Combine billing and payment info into shape (samples, time_steps=5, features=2)
X = np.stack([X_bill, X_pay], axis=2)

# Target is the 6th payment amount (PAY_AMT6)
y = df['PAY_AMT6'].values.reshape(-1, 1)

# Scale inputs and target to [0,1] range
scaler_X = MinMaxScaler()
X_reshaped = X.reshape(-1, 2)  # flatten time steps and features for scaler
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(-1, 5, 2)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 2)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate model on test set
loss = model.evaluate(X_test, y_test)
print(f"Test MSE Loss: {loss}")

# Make predictions and inverse scale
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Display sample predictions vs actual
for i in range(5):
    print(f"Predicted PAY_AMT6: {y_pred[i][0]:.2f} vs Actual: {y_test_actual[i][0]:.2f}")
