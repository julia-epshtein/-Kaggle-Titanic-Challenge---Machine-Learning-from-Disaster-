import numpy as np
import pandas as pd
import jax
from jax import grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

datapath = "./"

train_data = pd.read_csv('HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/train.csv')
test_data = pd.read_csv('HW 9/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-/test.csv')

col_names = train_data.columns

# Fix-shape universal approximator method (kernel method) 
def run_universal_approximator_method():
    x=2
    
# Neural network based on universal approximator method
def run_neural_network_based_method():
    x=2

################## TASK 3 ##################

# Tree-based approach
def run_tree_based_approach():
    # Obtain Training Set From Training Data and Testing Set From Testing Data
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)

# Main method
if __name__ == '__main__':
    #run_universal_approximator_method()
    #run_neural_network_based_method()
    run_tree_based_approach()