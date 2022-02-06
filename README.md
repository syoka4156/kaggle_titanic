# kaggle_titanic

## Overview
This is a tutorial for the Kaggle competition, [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview).
I tried four model patterns, and the best public score was 0.76794.

## Method
- Feature Extraction
  -  Embarked
          -  convert into dummy variable
  -  Survival_Rate
          - with binning Age
  -  with_Family 
          - calcualted from SibSp and Parch
- Under Sampling
  - RandomUnderSampler
  - ClusterCentroids 
- Hyper Parameter Optimization
  - Optuna
- Mathine Learning
  - Logistic Regression
  - Randomforest
- Cross Varidation
  - kFold (k=5)  

### Target
- Survived

### Features
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked 
- Survival_Rate 
- with_Family 

## Model
I made four patterns bellow.
|     | Under Sampling     | Model                  | 
| --- | ------------------ | ---------------------- | 
| 1   | RandomUnderSampler | RandomForestClassifier | 
| 2   | RandomUnderSampler | LogisticRegression     | 
| 3   | ClusterCentroids   | RandomForestClassifier | 
| 4   | ClusterCentroids   | LogisticRegression     | 

The following is the hyper parameters optimized by Optuna. 
|     | Hyper Parameter                                                                                             | Valid Accuracy | Test Accuracy | 
| --- | ----------------------------------------------------------------------------------------------------------- | -------------- | ------------- | 
| 1   | {'max_depth': 18, 'min_samples_split': 16, 'min_samples_leaf': 3, 'n_estimators': 800, 'random_state': 101} | 0.80           | 0.83          | 
| 2   | {'C': 52, 'solver': 'saga', 'penalty': 'l2', 'max_iter': 700, 'random_state': 101}                          | 0.79           | 0.83          | 
| 3   | {'max_depth': 7, 'min_samples_split': 16, 'min_samples_leaf': 1, 'n_estimators': 800, 'random_state': 101}  | 0.78           | 0.84          | 
| 4   | {'C': 4, 'solver': 'saga', 'penalty': 'l1', 'max_iter': 700, 'random_state': 101}                           | 0.74           | 0.82          | 

## Result
The public scores of models of ClusterCentroids are best. 
Those of RandomUnderSampler are a bit more than the scores of train data.
There might be overfitting.
|     | Accuracy | Precision | Recall | F1   | Public Score | 
| --- | -------- | --------- | ------ | ---- | ------------ | 
| 1   | 0.80     | 0.81      | 0.77   | 0.79 | 0.75598      | 
| 2   | 0.79     | 0.80      | 0.77   | 0.78 | 0.75598      | 
| 3   | 0.78     | 0.83      | 0.70   | 0.76 | 0.76794      | 
| 4   | 0.75     | 0.75      | 0.75   | 0.75 | 0.76794      | 
