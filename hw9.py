import pandas as pd
import numpy as np
import random as rnd

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Deep learning libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

datapath = "./"

train_data = pd.read_csv('HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/train.csv')
test_data = pd.read_csv('/Users/juliaepshtein/Desktop/CS589/HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/test.csv')
col_names = train_data.columns

################## TASK 0 ##################

# Handle Missing Values
train_data.fillna(value={'Age': train_data['Age'].median(), 'Fare': train_data['Fare'].mean()}, inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Convert Categorical Data
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data = pd.get_dummies(train_data, columns=['Embarked'])  # One-hot encode 'Embarked'

# Feature Engineering
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

# Drop Unnecessary Columns
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Normalize/Scale Numerical Data (using Standard Normalization)
train_data[['Age', 'Fare']] = StandardScaler().fit_transform(train_data[['Age', 'Fare']])

# Verify Data Types
train_data['Age'] = train_data['Age'].astype(float)

# Display the preprocessed training data
print(train_data.head())

# Save PassengerId column from test_data
passenger_ids_test = test_data['PassengerId']

# Preprocess test data
# Handle Missing Values
test_data.fillna(value={'Age': test_data['Age'].median(), 'Fare': test_data['Fare'].mean()}, inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Convert Categorical Data
test_data['Sex'] = label_encoder.transform(test_data['Sex'])
test_data = pd.get_dummies(test_data, columns=['Embarked'])  # One-hot encode 'Embarked'

# Feature Engineering
test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

# Drop Unnecessary Columns
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Normalize/Scale Numerical Data (using Min-Max scaling)
test_data[['Age', 'Fare']] = StandardScaler().fit_transform(test_data[['Age', 'Fare']])

# Verify Data Types
test_data['Age'] = test_data['Age'].astype(float)

# Table for model comparison
model_comparison = pd.DataFrame(columns=['Model', 'Accuracy'])

################## TASK 1 ##################

# Normalize the data with standard scaling

# Fix-shape universal approximator method (kernel method) 
def run_universal_approximator_method():
    x_train = train_data.drop(['Survived'], axis=1)
    y_train = train_data['Survived']
    x_test = test_data  
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    svc = SVC()
    svc.fit(x_train_scaled, y_train)
    Y_pred = svc.predict(x_test_scaled)
    acc_svc = round(svc.score(x_train_scaled, y_train) * 100, 2)
    print(acc_svc)
    
    predictions = Y_pred
    output = pd.DataFrame({'PassengerId': passenger_ids_test, 'Survived': predictions})
    output.to_csv('submission_svc.csv', index=False)
    
    acc_svc = round(svc.score(x_train_scaled, y_train) * 100, 2)
    model_comparison.loc[0] = ['SVM', acc_svc]


################## TASK 2 ##################

def run_neural_network_based_method():
    x_train = train_data.drop(['Survived'], axis=1)
    y_train = train_data['Survived']
    y_train = y_train.astype('float32')  
    x_test = test_data  
    
    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    model.fit(x_train, np.array(y_train), epochs=100)

    predictions = (model.predict(x_test) > 0.5).astype(int).ravel()
    output = pd.DataFrame({'PassengerId': passenger_ids_test, 'Survived': predictions}) 
    output.to_csv('submission_nn.csv', index=False)
    
    accuracy_nn = model.evaluate(x_train, y_train)[1] * 100
    model_comparison.loc[1] = ['Neural Network', accuracy_nn]

################## TASK 3 ##################

def run_tree_based_approach():
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print(f"Accuracy: {accuracy}")

    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission_trees.csv', index=False)
    
    accuracy_tree = metrics.accuracy_score(y_valid, y_pred) * 100
    model_comparison.loc[2] = ['Random Forest', accuracy_tree]


# Main method
if __name__ == '__main__':
    run_universal_approximator_method()
    run_neural_network_based_method()
    run_tree_based_approach()
    print(model_comparison)

