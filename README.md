# Bank term deposit subscription prediction

## Problem
The goal of this project is to predict whether a client will subscribe to a term deposit based on client data, previous campaign interactions, and macroeconomic indicators. This is a binary classification problem with a strong class imbalance (approximately 11% positive class).

## Data

The dataset does not contain missing values. Some categorical variables include an "unknown" category, which was preserved as it may carry useful information. The target variable is highly imbalanced, with the majority of observations belonging to the negative class

## Approach

All steps are implemented in a single Jupyter Notebook, including exploratory data analysis, feature engineering, encoding of categorical variables, model training, threshold tuning, SHAP-based interpretation, and error analysis

## Models

The following models were trained and compared: Logistic Regression, k-Nearest Neighbors, Decision Tree, and XGBoost (base model, RandomizedSearchCV, and Hyperopt tuning).

## Evaluation

Due to class imbalance, model performance was evaluated using ROC-AUC and F1-score.

## Model Comparison

| model | hyperparameters | train_roc_auc | valid_roc_auc | train_f1 | valid_f1 | comment |
|------|----------------|--------------|--------------|----------|----------|---------|
| Logistic Regression | class_weight=balanced, max_iter=1000 | 0.795 | 0.801 | 0.452 | 0.469 | Good baseline model with stable performance. Can be improved with threshold tuning and feature engineering |
| kNN | n_neighbors=6 | 0.916 | 0.755 | 0.523 | 0.437 | Shows overfitting and lower performance than Logistic Regression. Possible improvements include tuning hyperparameters and resampling |
| Decision Tree | max_depth=5, class_weight=balanced | 0.788 | 0.796 | 0.469 | 0.481 | Stable model with no clear overfitting and slightly better F1 than Logistic Regression |
| XGBoost base | n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8 | 0.869 | 0.811 | 0.486 | 0.395 | High ROC-AUC but low recall at default threshold. Threshold tuning is required |
| XGBoost Randomized Search + threshold | subsample=0.9, n_estimators=300, min_child_weight=5, max_depth=5, learning_rate=0.01, gamma=1, colsample_bytree=1.0, threshold=0.2 | 0.82 | 0.815 | 0.509 | 0.526 | Strong model with improved F1 after threshold tuning |
| XGBoost Hyperopt + threshold | colsample_bytree=0.6, gamma=0.6, learning_rate=0.02, max_depth=6, min_child_weight=6, n_estimators=200, subsample=0.7, threshold=0.2 | 0.835 | 0.816 | 0.512 | 0.531 | Best performing model with balanced precision and recall |

## Interpretation

Feature importance and SHAP analysis show that macroeconomic variables such as `nr.employed`, `emp.var.rate`, and `euribor3m` have the strongest impact on predictions. Campaign-related features (e.g. number of contacts) and previous interactions also influence the outcome.

## Error Analysis

Analysis of misclassified cases shows that the model tends to rely heavily on macroeconomic signals. False negatives occur when strong negative macro and campaign signals override weaker positive indicators. False positives appear when favorable economic conditions lead the model to overestimate the probability of subscription, even if individual behavior does not support it.

## Conclusions

XGBoost achieved the best overall performance. Threshold tuning had a significant impact on F1-score, while differences between tuning methods were relatively small. The model performs well but struggles in cases where macroeconomic signals conflict with individual behavior.

## Future Work

Possible improvements include adding more customer-level behavioral features, introducing feature interactions, reducing reliance on macroeconomic variables, applying calibration techniques, and exploring ensemble approaches.
