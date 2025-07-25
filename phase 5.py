import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
df = pd.read_csv(r"C:\Users\SaiKrishna\OneDrive\Desktop\UCI_Credit_Card.csv")

# Time series columns (5 months)
bill_cols_5 = [f'BILL_AMT{i}' for i in range(1, 6)]
pay_cols_5 = [f'PAY_AMT{i}' for i in range(1, 6)]

# Static features
static_cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE']

# Prepare time series data
X_bill = df[bill_cols_5].values
X_pay = df[pay_cols_5].values
X_time = np.stack([X_bill, X_pay], axis=2)  # shape (samples, 5, 2)

# Prepare static features
X_static = df[static_cols].values

# Target
y = df['default.payment.next.month'].values

# Scale time series features (MinMax)
scaler_time = MinMaxScaler()
X_time_reshaped = X_time.reshape(-1, 2)
X_time_scaled = scaler_time.fit_transform(X_time_reshaped).reshape(-1, 5, 2)

# Scale static features (StandardScaler)
scaler_static = StandardScaler()
X_static_scaled = scaler_static.fit_transform(X_static)

# Split data
X_time_train, X_time_test, X_static_train, X_static_test, y_train, y_test = train_test_split(
    X_time_scaled, X_static_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Build model

# Time series input branch
input_time = tf.keras.Input(shape=(5, 2), name='time_series_input')
x1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(input_time)
x1 = tf.keras.layers.Dropout(0.3)(x1)

# Static input branch
input_static = tf.keras.Input(shape=(len(static_cols),), name='static_input')
x2 = tf.keras.layers.Dense(32, activation='relu')(input_static)
x2 = tf.keras.layers.Dropout(0.3)(x2)

# Combine
combined = tf.keras.layers.concatenate([x1, x2])
x = tf.keras.layers.Dense(64, activation='relu')(combined)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_time, input_static], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    {'time_series_input': X_time_train, 'static_input': X_static_train},
    y_train,
    epochs=25,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=2
)

# Evaluate
test_loss, test_acc = model.evaluate({'time_series_input': X_time_test, 'static_input': X_static_test}, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Predict
y_pred_prob = model.predict({'time_series_input': X_time_test, 'static_input': X_static_test})
y_pred = (y_pred_prob > 0.5).astype(int)

# Detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")
