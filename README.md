# Titanic-Survival-Prediction
A machine learning model to predict Titanic survival based on various passenger attributes

# Link to the data set

https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/tested.csv

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

# Import necessary libraries
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt

import seaborn as sns

# Load the Titanic dataset (replace with your file path if needed)

df = pd.read_csv('D:\\tested.csv')

# 1. Data Preprocessing

# Drop unnecessary columns

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values (mean for age, mode for Embarked, drop rows with missing Survived)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.dropna(subset=['Survived'], inplace=True)

# Encode categorical variables (Sex, Embarked)

label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male=1, Female=0

df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # C=0, Q=1, S=2

# 2. Feature Selection (target is 'Survived')

X = df.drop('Survived', axis=1)

y = df['Survived']

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# 5. Model Training

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_scaled, y_train)

# 6. Model Prediction

y_pred = model.predict(X_test_scaled)

# 7. Evaluation

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# 8. Visualization of Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix')

plt.show()  
    
# Project Dependencies
To Add a requirements.txt file to list all the dependencies.
Example of requirements.txt:

pandas==1.5.3

numpy==1.24.1

scikit-learn==1.2.1

matplotlib==3.6.2

seaborn==0.11.2

# Link to see the images

https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/Screenshot%20(57).png

https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/Screenshot%20(58).png

https://github.com/kalyani12345121/Titanic-Survival-Prediction/blob/main/Screenshot%20(59).png
