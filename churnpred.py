#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use("fivethirtyeight")

# %%
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head(3)
df.shape
df.isnull()
df.info()

# %%
df.columns

df['gender'].value_counts()
df['SeniorCitizen'].value_counts()
df['SeniorCitizen'].unique()

for col in df.columns:
    if col not in ["tenure", "MonthlyCharges"]:
        print(col, df[col].unique())
        print('----------------------------------------')


#%%
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    print(col, len(df[df[col]== " "]))
    print("-------------------------------------")


df['TotalCharges'] = df['TotalCharges'].replace({" ": "0.0"}).astype(float)

# %%

def plot_distribution(df, column_name):
    plt.figure(figsize= (12, 4) )
    plt.subplot(1, 2, 1)
    sns.histplot(df[column_name], kde= True)
    plt.title(f"Distribution of {column_name}")

    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    plt.axvline(col_mean, color = "red", linestyle = '--', label = "Mean")
    plt.axvline(col_median, color = "red", linestyle = '--', label = "Median")

    plt.legend()

    plt.subplot(1, 2, 2)
    sns.boxplot(y = df[column_name])
    plt.title(f"Boxplot of {column_name}")
    plt.show()


# %%

plot_distribution(df, "tenure")
plot_distribution(df, "MonthlyCharges")

# %%

plot_distribution(df, "TotalCharges")

# %%

plt.figure(figsize = (8, 4))
sns.heatmap(df[['tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap = "coolwarm", fmt = ".2f")
plt.title("Correlation Matrix")
plt.show()


# %%

categorical_cols = df.select_dtypes(include = "object").columns.to_list() + ['SeniorCitizen']

for col in categorical_cols:
    plt.figure(figsize = (6, 4))
    sns.countplot(data = df, x = col, hue = 'Churn')
    plt.title(f"{col} Distribution by Churn")
    plt.show()
# %%
df['Churn'] = df["Churn"].replace({'Yes' : 1, 'No' :0})
df.info()

# %%
object_columns = df.select_dtypes(include="object").columns

object_columns 

# %%
from sklearn.preprocessing import LabelEncoder

encoders = {}

for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] =  label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder
# %%

encoders
df.info()

# %%

import pickle
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)


# %%
from sklearn.preprocessing import StandardScaler

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# %%
df["Churn"].value_counts()


# %%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# %%
X = df.drop(columns= ["Churn"])
Y = df["Churn"]

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42)

Y_train.value_counts()


# %%
smote = SMOTE(random_state= 42)

X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)

Y_train_smote.value_counts()
# %%
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}
# %%
hyperparameters = {
    "Random Forest" :{
        "n_estimators": [50,100, 200],
        "max_depth": [5, 10, None],
    },
    "XGBoost":{
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    }
}
# %%
best_models = {}
for model_name, model in models.items():
    grid = GridSearchCV(estimator=model, param_grid=hyperparameters[model_name], cv=5, scoring = "accuracy")
    grid.fit(X_train_smote, Y_train_smote)
    best_models[model_name] = grid.best_estimator_
    print(f"Best parameters for {model_name} : {grid.best_params_}")
    print(f"Best Accuracy for {model_name} : {grid.best_score_: .2f}\n")
# %%
best_models

# %%

with open("best_models.pkl", "wb") as f:
    pickle.dump(best_models['Random Forest'], f)
# %%
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# %%
y_test_pred = best_models["Random Forest"].predict(X_test)
y_test_prob = best_models["Random Forest"].predict_proba(X_test)[:, 1]
#%%
y_test_pred

# %%
y_test_prob
# %%
print(f"Accuracy : ", accuracy_score(Y_test, y_test_pred))
print(f"ROC - AUC Score : ", roc_auc_score(Y_test, y_test_pred))
print(f"Confusion Matrix : \n", confusion_matrix(Y_test, y_test_pred))
print(f"Classification Report : \n", classification_report(Y_test, y_test_pred))
# %%
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler_data =pickle.load(f)


# %%
def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    return "Churn" if prediction == 1 else "No Churn", probability

#%%
df.columns

#%%
example_input = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

#%%
prediction, prob = make_prediction(example_input)

#%%
print(f"Prediction: {prediction}, Probability : {prob : .2f}")