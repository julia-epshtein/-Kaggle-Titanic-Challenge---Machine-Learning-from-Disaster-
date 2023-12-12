# Titanic Survival Prediction Project
# Titanic Survival Prediction Project

## Setting Up and Running the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/julia-epshtein/-Kaggle-Titanic-Challenge---Machine-Learning-from-Disaster-.git
   cd titanic-survival-prediction
   ```

2. **Install Dependencies:**
   ```bash
   ## To install Jax follow these steps:
   - Create a new Conda environment: conda create -n jax_env
   - Activate the new environment: conda activate jax_env
   - Install Jax in your Conda environment using the following command: conda install -c conda-forge jax
   - Install matplotlib in your environment using: conda install -c conda-forge matplotlib
   
   ## NumPy and Pandas
   pip install numpy
   
   ## Tensorflow
   pip install tensorflow
   ```

3. **Run the Jupyter Notebooks:**
   - Open and run the Jupyter notebooks in the `notebooks` directory for data preprocessing and model implementation.

4. **Run the Python Scripts:**
   - Execute the Python scripts in the `scripts` directory for specific tasks, such as data preprocessing and model training.

5. **Explore Results:**
   - Check the accuracy table in the README or explore the generated visualizations and predictions.

6. **Submit Predictions to Kaggle:**
   - If desired, follow the Kaggle submission guidelines provided in the notebooks or scripts to submit predictions.

## Introduction

The Titanic Survival Prediction project focuses on developing a machine learning model to predict passenger survival on the Titanic based on various features. The dataset includes information about passengers, such as age, sex, ticket class, fare, embarked port, and cabin information. The primary objective is to create a predictive model capable of accurately determining whether a passenger survived or not.

## Methodology

### Data Preprocessing

- **Handling Missing Values:** Imputed missing values in the 'Age' and 'Fare' columns with median and mean values, respectively. The 'Embarked' column was filled with the mode.
  
- **Encoding Categorical Data:** Label-encoded the 'Sex' column and applied one-hot encoding to the 'Embarked' column.
  
- **Feature Engineering:** Created a new binary feature, 'Has_Cabin,' to indicate whether a passenger had a cabin. Dropped unnecessary columns ('Name', 'Ticket', 'Cabin').
  
- **Normalization/Scaling:** Normalized numerical data ('Age' and 'Fare') using Standard Scaling.

### Model Implementation

#### Learning Approaches Implementation

**Task 1: Fix-Shape Universal Approximator (SVM)**
- Standardized data using Standard Scaling.
- Utilized a Support Vector Machine (SVC) classifier.
- Achieved an accuracy of approximately [acc_svc]% on the training set.

**Task 2: Neural Network-Based Method**
- Used data as is.
- Implemented a neural network with multiple layers using Keras.
- Trained the model for 100 epochs.

**Task 3: Tree-Based Approach (Random Forest)**
- One-hot encoded selected features.
- Split data into training and validation sets.
- Trained a Random Forest classifier with specified hyperparameters.
- Calculated the accuracy on the validation set.

## Results

### Design Justification

SVM and neural network methods were chosen for their ability to capture complex patterns, while Random Forest was selected for its robustness and ease of implementation.

### Optimal Parameters

- **SVM:** Standard scaling was used, and no additional hyperparameter tuning was performed.
- **Neural Network:** Multiple layers with ReLU activation, binary cross-entropy loss, and Adam optimizer were used.
- **Random Forest:** 100 estimators, max depth of 10 were chosen.

### Metrics

Accuracy was the evaluation metric for SVM, neural network, and Random Forest on the validation set, and Kaggle was used to evaluate performance on the test set.

### Visualizations

#### Accuracy Table

| Model Name      | Model Accuracy |
| --------------- | --------------- |
| SVM             | 84.74%          |
| Neural Network  | 80.36%          |
| Random Forest   | 75.42%          |

