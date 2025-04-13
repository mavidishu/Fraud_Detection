from flask import Flask, request, render_template
import os
import joblib
from dotenv import load_dotenv
import pandas as pd
import datetime

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route("/fraud")
def fraud_page():
    return render_template("fraud.html")

model = joblib.load('model_output/random_forest_model.pkl')
label_mappings = joblib.load('model_output/label_mappings.pkl')
# def preprocess_input(form_data):
#     # Convert form data into a dataframe
#     df = pd.DataFrame([form_data])
    
#     # Feature Engineering
#     df["payerdebited"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
#     df["recievercredited"] = df["newbalanceDest"] - df["oldbalanceDest"]
#     df["reciever_type"] = df["nameDest"].str[0]
#     df["payer_type"] = df["nameOrig"].str[0]

#     # Drop unnecessary fields
#     drop_cols = ["nameDest", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
#                  "newbalanceDest", "oldbalanceDest", "step", "payer_type"]
#     df.drop(columns=drop_cols, inplace=True)

#     # Encode categorical
#     for col, mapping in label_mappings.items():
#         df[col] = df[col].map(mapping).fillna(0).astype(int)
    
#     # Drop recievercredited to match training
#     df.drop(columns=["recievercredited"], inplace=True)

#     return df

def preprocess_input(form_data):
    # Convert form data into a dataframe
    df = pd.DataFrame([form_data])
    
    # Generate date, day_of_week, and hour from 'step'
    base_date = datetime.datetime(2024, 1, 1)  # adjust base date if needed
    df["date"] = base_date + pd.to_timedelta(df["step"], unit='h')
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour

    # Feature Engineering
    df["payerdebited"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["recievercredited"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["reciever_type"] = df["nameDest"].str[0]
    df["payer_type"] = df["nameOrig"].str[0]
    df['date'] = (pd.to_datetime('2024-04-01') + pd.to_timedelta(df['step'], unit='h')).dt.day
    # Drop unnecessary fields
    drop_cols = ["nameDest", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
                 "newbalanceDest", "oldbalanceDest", "step", "payer_type"]
    df.drop(columns=drop_cols, inplace=True)

    # Encode categorical
    for col, mapping in label_mappings.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    # Drop recievercredited if it wasn't used during training
    df.drop(columns=["recievercredited"], inplace=True, errors="ignore")

    # Reorder columns to match model's expected input
    expected_features = list(model.feature_names_in_)
    print(expected_features)
    df = df[expected_features]
    print(df)
    return df

@app.route('/result', methods=['GET', 'POST'])
def result():
    print(request.method) # Debugging line to check request method
    print(request.form)
    if request.method == 'POST':
        # Get form input
        input_data = {
            'step': int(request.form['step']),
            'type': request.form['type'],
            'amount': float(request.form['amount']),
            'nameOrig': request.form['nameOrig'],
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'nameDest': request.form['nameDest'],
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest']),
        }
        print(input_data) # Debugging line to check input data
        df = preprocess_input(input_data)
        amount = input_data['amount']
        pred = model.predict(df)[0]

        def classify(amount, prediction):
            return "Fraudulent" if amount > 200000 or prediction == 1 else "Legit"

        result = classify(amount, pred)
        print("Result of the model:",result)
        return render_template('fraud.html', result=result)

    return render_template('fraud.html')

if __name__ == "__main__":
    app.run(debug=True) # Ensures auto-reloading