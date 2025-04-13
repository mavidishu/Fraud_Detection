import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load data
file_path = r"C:\Dishu Mavi\.amity_subjects\Sem 8\NTCC\archive\Fraud.csv"
df = pd.read_csv(file_path)

# Feature Engineering
df["payerdebited"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["recievercredited"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["reciever_type"] = df["nameDest"].str[0:1]
df["payer_type"] = df["nameOrig"].str[0:1]
df['datetime'] = pd.to_datetime('2024-04-01') + pd.to_timedelta(df['step'], unit='h')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['date'] = df['datetime'].dt.day

# Drop irrelevant columns
df.drop(columns=[
    "nameDest", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "newbalanceDest", "oldbalanceDest", "payer_type", "step",
    "datetime", "isFlaggedFraud"
], inplace=True)

# Encode categorical variables
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
encoder = LabelEncoder()
encoded_df = df.copy()
mappings = {}

for col in cat_cols:
    encoded_df[col] = encoder.fit_transform(df[col])
    mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Drop unnecessary column
encoded_df.drop(columns=["recievercredited"], inplace=True)

# Prepare features and custom label
X = encoded_df.drop(columns="isFraud")
y = pd.Series([
    1 if encoded_df.loc[i, "amount"] > 200000 else encoded_df.loc[i, "isFraud"]
    for i in range(len(encoded_df))
])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=15, class_weight='balanced', criterion='entropy', random_state=42
)
model.fit(x_train, y_train)

# Prediction with fraud logic
def illegal_transac(amount, model_prediction):
    return 1 if amount > 200000 else model_prediction

raw_preds = model.predict(x_test)
y_pred = [illegal_transac(amount, pred) for amount, pred in zip(x_test['amount'], raw_preds)]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and mappings
output_dir = "./model_output"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, "random_forest_model.pkl"))
joblib.dump(mappings, os.path.join(output_dir, "label_mappings.pkl"))

print(f"\n💾 Model and mappings saved in: {output_dir}")