import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv(r"C:\Users\SaiKrishna\OneDrive\Desktop\UCI_Credit_Card.csv")

# Time series features
bill_cols_5 = [f'BILL_AMT{i}' for i in range(1, 6)]
pay_cols_5 = [f'PAY_AMT{i}' for i in range(1, 6)]

X_bill = df[bill_cols_5].values
X_pay = df[pay_cols_5].values
X = np.stack([X_bill, X_pay], axis=2)  # shape: (samples, 5, 2)

# Target - binary classification for default next month
y = df['default.payment.next.month'].values

# Scale input features
scaler = MinMaxScaler()
X_reshaped = X.reshape(-1, 2)
X_scaled = scaler.fit_transform(X_reshaped).reshape(-1, 5, 2)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build LSTM classification model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, 2)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predict and classification report
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
