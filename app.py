from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model, encoders, and scaler
# Note: Renaming 'best_models.pkl' to 'best_model.pkl' to match your
# churnpred.py saving (you saved 'best_models.pkl' but loaded 'best_model.pkl')
# Let's assume the correct file is 'best_models.pkl' as in your original app.py
try:
    with open('best_models.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    # Fallback in case you named it 'best_model.pkl'
    with open('best_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = Flask(__name__)

# Define the exact list of features and their order from churnpred.py
# This is (df.columns.drop('Churn'))
MODEL_FEATURES = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# Define which columns were scaled
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def make_prediction(input_data):
    # 1. Create a DataFrame from the form input
    input_df = pd.DataFrame([input_data])

    # 2. Handle all categorical (encoded) features
    for col, encoder in encoders.items():
        if col == 'customerID':
            # Add the dummy 'customerID' column that the model
            # was incorrectly trained on. We'll use 0 as a
            # placeholder encoded value.
            input_df['customerID'] = 0
        else:
            # Transform all other categorical columns from the form
            input_df[col] = encoder.transform(input_df[col])

    # 3. Handle all numerical (scaled) features
    input_df[NUMERICAL_COLS] = scaler_data.transform(input_df[NUMERICAL_COLS])

    # 4. Re-order the DataFrame to match the exact order the model was trained on
    final_input_df = input_df[MODEL_FEATURES]

    # 5. Make the prediction
    prediction = loaded_model.predict(final_input_df)[0]
    probability = loaded_model.predict_proba(final_input_df)[0, 1]
    
    return "Churn" if prediction == 1 else "No Churn", probability

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        # Get TotalCharges and handle empty string (like in churnpred.py)
        total_charges_input = request.form['TotalCharges']
        total_charges = float(total_charges_input) if total_charges_input.strip() else 0.0

        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': total_charges,
        }

        prediction, probability = make_prediction(input_data)

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)