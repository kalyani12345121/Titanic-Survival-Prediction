# Titanic-Survival-Prediction
A machine learning model to predict Titanic survival based on various passenger attributes

# Structure
Titanic-Survival-Prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore

# Task Objectives:
To Build a machine learning model to predict survival on the Titanic using the dataset of passengers.
I Implement steps like data cleaning, feature engineering, model building, and evaluation.
# steps to run the project
https://github.com/kalyani12345121/Titanic-Survival-Prediction/edit/main/README.md

# Load the dataset and start preprocessing:
Open notebooks/data_preprocessing.ipynb to start data preprocessing.
# Train the model:
 To Open notebooks/model_training.ipynb to train the model.
# Evaluate the model:
To Run the model evaluation script src/evaluate.py.
Clean and Well-Structured Code:
Code will be modular and separated into distinct sections for preprocessing, model training, and evaluation.

# Implementation Steps
1. Data Preprocessing
Objective: Clean the data by handling missing values, encoding categorical variables, and normalizing numerical data.
Steps:
Load train.csv and test.csv files.
Handle missing values (e.g., fill missing age with mean or median, drop or fill missing Embarked).
To Encode categorical features like Sex, Embarked using one-hot encoding or label encoding.
Normalize numerical features like Age, Fare (e.g., using MinMaxScaler or StandardScaler).
Code Example (in src/preprocessing.py):
python
Copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
      df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
          scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    return df
    
# Model Training
Objective: Build a classification model to predict whether a passenger survived the Titanic disaster.
Steps:
Split data into training and validation sets.
Train various machine learning models (e.g., Logistic Regression, Random Forest, XGBoost).
Evaluate models using performance metrics like accuracy, precision, recall, F1-score.
Code Example (in src/model.py):
python
Copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import preprocess_data

def train_model(data_path):
    df = preprocess_data(data_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
           print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Model Evaluation
Objective: Evaluate the trained model and provide insights on performance.
# Steps:
Evaluate performance on test data using accuracy, precision, recall, F1-score.
Analyze the confusion matrix.
Code Example (in src/evaluate.py):
python
Copy
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")  
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
# Project Dependencies
To Add a requirements.txt file to list all the dependencies.
Example of requirements.txt:
ini
Copy
pandas==1.5.3
numpy==1.24.1
scikit-learn==1.2.1
matplotlib==3.6.2
seaborn==0.11.2

# Link to see the images

https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/Screenshot%20(57).png
https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/Screenshot%20(58).png
