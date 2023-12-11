# Data manipulation libraries
import pandas as pd
import numpy as np
import random as rnd

# Data visualization libraries
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Deep learning libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

# JAX library for automatic differentiation
import jax
from jax import grad
import jax.numpy as jnp

datapath = "./"

train_data = pd.read_csv('HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/train.csv')
test_data = pd.read_csv('HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/test.csv')
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
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Normalize/Scale Numerical Data (using Standard Normalization)
train_data[['Age', 'Fare']] = (train_data[['Age', 'Fare']] - train_data[['Age', 'Fare']].mean()) / train_data[['Age', 'Fare']].std()

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
test_data[['Age', 'Fare']] = (test_data[['Age', 'Fare']] - test_data[['Age', 'Fare']].min()) / (test_data[['Age', 'Fare']].max() - test_data[['Age', 'Fare']].min())

# Verify Data Types
test_data['Age'] = test_data['Age'].astype(float)

################## TASK 1 ##################

# Normalize the data with standard scaling

# Fix-shape universal approximator method (kernel method) 
def run_universal_approximator_method():    
    x_train = train_data.drop(['Survived'], axis = 1)
    y_train = train_data['Survived']
    x_test = test_data.drop(['PassengerId'], axis = 1)
    y_test = test_data['PassengerId']
    
    svc = SVC()
    svc.fit(x_train, y_train)
    Y_pred = svc.predict(x_test)
    acc_svc = round(svc.score(x_train, y_train) * 100, 2)
    print(acc_svc)



################## TASK 2 ##################

def run_neural_network_based_method():
    x_train = train_data.drop(['Survived'], axis = 1)
    y_train = train_data['Survived']
    x_test = test_data.drop(['PassengerId'], axis = 1)
        
    model = Sequential()
    model.add(Dense(units = 32, input_shape = (7,), activation = 'relu'))
    model.add(Dense(units = 64, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(units = 128, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
    model.add(Dropout(0.1))
    model.add(Dense(units = 64, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
    model.add(Dropout(0.1))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(units = 16, activation = 'relu'))
    model.add(Dense(units = 8, activation = 'relu',kernel_initializer = 'he_normal', use_bias = False))
    model.add(Dense(units =1 , activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['acc'])    
    model.fit(x_train, y_train, batch_size = 32, verbose = 2, epochs = 50)
    
    # predict = model.predict(x_test)
    # #since we have use sigmoid activation function in output layer
    # predict = (predict > 0.5).astype(int).ravel()
    # print(predict)
    
    # y_pred_rand = (model.predict(x_train) > 0.5).astype(int)
    # print('Accuracy : ', np.round(metrics.accuracy_score(y_train, y_pred_rand)*100,2))  

    
################## TASK 3 ##################

def run_tree_based_approach():
    
    # Obtain Training Set From Training Data and Testing Set From Testing Data
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

    # Create and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_valid)

    # Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save Submission
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    
# Main method
if __name__ == '__main__':
    run_universal_approximator_method()
    #run_neural_network_based_method()
    #run_tree_based_approach()