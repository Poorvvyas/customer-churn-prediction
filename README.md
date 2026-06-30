# Customer Churn Prediction System

A machine learning-based web application that predicts customer churn for telecommunications companies. The system uses ensemble learning techniques to identify customers at risk of leaving, enabling proactive retention strategies.

## 🎯 Project Overview

This project implements a complete end-to-end machine learning pipeline for predicting customer churn, including data preprocessing, model training, hyperparameter tuning, and deployment through a Flask web application.

## ✨ Features

- **Interactive Web Interface**: User-friendly form for inputting customer data
- **Real-time Predictions**: Instant churn probability assessment
- **Multiple ML Models**: Comparison between Random Forest and XGBoost classifiers
- **Imbalanced Data Handling**: SMOTE technique for balanced training
- **Hyperparameter Optimization**: GridSearchCV for optimal model performance
- **Bootstrap UI**: Clean and responsive design

## 🛠️ Technologies Used

- **Python 3.x**
- **Machine Learning**: scikit-learn, XGBoost, imbalanced-learn
- **Web Framework**: Flask
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, Bootstrap 5

## 📊 Dataset

The project uses the Telco Customer Churn dataset containing customer information including:
- Demographics (gender, senior citizen status, partner, dependents)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, security)
- Billing information (monthly charges, total charges)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📦 Required Packages

Create a `requirements.txt` file with:
```
flask==3.0.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.0.2
imbalanced-learn==0.11.0
matplotlib==3.8.2
seaborn==0.13.0
```

## 🏃 Running the Application

1. Train the model (if not already trained):
```bash
python churnpred.py
```

This will generate:
- `best_models.pkl`: Trained Random Forest model
- `encoder.pkl`: Label encoders for categorical features
- `scaler.pkl`: StandardScaler for numerical features

2. Start the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py                              # Flask web application
├── churnpred.py                        # Model training and evaluation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt                    # Python dependencies
│
├── templates/
│   └── index.html                      # Web interface
│
├── best_models.pkl                     # Trained model
├── encoder.pkl                         # Fitted label encoders
├── scaler.pkl                          # Fitted scaler
│
└── README.md                           # Project documentation
```

## 🔍 Model Performance

The Random Forest classifier achieved:
- **Accuracy**: ~80%
- **ROC-AUC Score**: Evaluated on test set
- **Training**: Used SMOTE for handling class imbalance
- **Validation**: 5-fold cross-validation

## 💡 Usage

1. Fill in the customer information in the web form
2. Click "Predict" button
3. View the prediction result and churn probability

Example input:
- Gender: Female
- Senior Citizen: No
- Partner: Yes
- Tenure: 12 months
- Contract: Month-to-month
- Monthly Charges: $70.00
- (and other relevant fields)

## 🔧 Model Training Details

The training pipeline includes:
1. **Data Preprocessing**: Handling missing values, encoding categorical variables
2. **Feature Scaling**: StandardScaler for numerical features
3. **Class Balancing**: SMOTE oversampling for minority class
4. **Model Selection**: GridSearchCV with Random Forest and XGBoost
5. **Evaluation**: Accuracy, ROC-AUC, confusion matrix, classification report

##  Acknowledgments

- Dataset source: Telco Customer Churn Dataset
- Built as part of machine learning portfolio project
