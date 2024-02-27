import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load the dataset, skipping the first row
data = pd.read_csv('PYTHON\\Blood Donation\\transfusion.data', skiprows=1, header=None)
data.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donation_March_2007']

# Split features and target variable
X = data.drop('Donation_March_2007', axis=1)
y = data['Donation_March_2007']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for numerical variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection and Training
# Using XGBoost classifier
model = xgb.XGBClassifier()
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
