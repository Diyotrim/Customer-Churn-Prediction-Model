from flask import Flask, render_template, request, redirect, url_for, send_file
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and tools
def load_model_and_tools():
    xgb_model = joblib.load('model/xgb_model.joblib')
    scaler = joblib.load('model/scaler.joblib')
    label_encoders = {
        'country': joblib.load('model/country_encoder.joblib'),
        'gender': joblib.load('model/gender_encoder.joblib')
    }
    return xgb_model, scaler, label_encoders

expected_features = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 
                     'products_number', 'credit_card', 'active_member', 'estimated_salary']

def convert_to_numeric(value):
    if isinstance(value, str):
        return 1 if value.lower() == 'yes' else 0
    return value

def predict_churn(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary):
    xgb_model, scaler, label_encoders = load_model_and_tools()

    # Convert categorical input data to numeric
    country_encoded = label_encoders['country'].transform([country])[0]
    gender_encoded = label_encoders['gender'].transform([gender])[0]

    # Convert Yes/No to 1/0
    credit_card_encoded = convert_to_numeric(credit_card)
    active_member_encoded = convert_to_numeric(active_member)

    # Create a feature array
    input_features = np.array([[credit_score, country_encoded, gender_encoded, age, tenure, balance, products_number, credit_card_encoded, active_member_encoded, estimated_salary]])
    
    # Standardize the features
    input_features_scaled = scaler.transform(input_features)
    
    # Predict using the trained XGBoost model
    prediction = xgb_model.predict(input_features_scaled)
    prediction_proba = xgb_model.predict_proba(input_features_scaled)[:, 1]
    
    return prediction[0], prediction_proba[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        credit_score = float(request.form['credit_score'])
        country = request.form['country']
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        products_number = int(request.form['products_number'])
        credit_card = request.form['credit_card']
        active_member = request.form['active_member']
        estimated_salary = float(request.form['estimated_salary'])

        # Predict churn
        prediction, prediction_proba = predict_churn(credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary)

        # Render result
        return render_template('result.html', prediction=prediction, prediction_proba=prediction_proba)
    
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read and preprocess dataset
            df = pd.read_csv(file)
            
            # Ensure expected features are in the dataset
            if not all(feature in df.columns for feature in expected_features):
                return "Dataset does not contain the required columns."

            # Load model and tools
            xgb_model, scaler, label_encoders = load_model_and_tools()

            # Preprocess data
            df_preprocessed = df[expected_features].copy()
            for column, le in label_encoders.items():
                if column in df_preprocessed.columns:
                    df_preprocessed[column] = df_preprocessed[column].apply(lambda x: x if x in le.classes_ else 'Unknown')
                    df_preprocessed[column] = le.transform(df_preprocessed[column])

            # Convert DataFrame to correct data types for prediction
            df_preprocessed = df_preprocessed.astype(float)

            # Standardize the features
            df_preprocessed_scaled = scaler.transform(df_preprocessed)

            # Make predictions
            predictions = xgb_model.predict(df_preprocessed_scaled)
            df['Churn Prediction'] = ["Will Churn" if p == 1 else "Will Not Churn" for p in predictions]

            # Save the results to a new CSV file
            result_file = 'churn_predictions.csv'
            df.to_csv(result_file, index=False)

            return redirect(url_for('download_file', filename=result_file))

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
