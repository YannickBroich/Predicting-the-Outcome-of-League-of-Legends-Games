# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:54:38 2025

@author: Yannick
"""

model_results = []

X = LolDataframe.drop(columns=['blueWins'])
y = LolDataframe['blueWins']  # Target variable

# Polynomial and interaction terms for LASSO
X_lasso = X.copy() # accuracy of .714
X_lasso['gold_xpdiff'] = X['blueGoldPerMin'] * X['blueExperienceDiff'] #.717
X_lasso['xpdiff.p2'] = X['blueExperienceDiff'] ** 2 #.720

## 0.1. Experience Difference Logistic Regression
print("\n--- ExperienceDiff Logistic Regression ---")

# train (45%), validation (45%), and test (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

xpdiff_model = LogisticRegressionCV(
    penalty='l1',
    solver='liblinear',
    cv=5,
    random_state=42,
    max_iter=1000
)
xpdiff_model.fit(X_train[['blueExperienceDiff']], y_train)

y_pred_xpdiff = xpdiff_model.predict(X_test[['blueExperienceDiff']])


xpdiff_accuracy = accuracy_score(y_test, y_pred_xpdiff)
print("\nAccuracy (XP Benchmark):", xpdiff_accuracy)


model_results.append({
    "Model": "ExperienceDiff Logistic Regression",
    "Accuracy": xpdiff_accuracy
})

## 0.2. Dragon Logistic Regression
print("\n--- Dragon Logistic Regression ---")
dragon_model = LogisticRegressionCV(
    penalty='l1',
    solver='liblinear',
    cv=5,
    random_state=42,
    max_iter=1000
)
dragon_model.fit(X_train[['blueDragons']], y_train)

y_pred_dragon = dragon_model.predict(X_test[['blueDragons']])


dragon_accuracy = accuracy_score(y_test, y_pred_dragon)
print("\nAccuracy (Dragon Benchmark):", dragon_accuracy)


model_results.append({
    "Model": "Dragon Logistic Regression",
    "Accuracy": dragon_accuracy
})

## 1. LASSO Logistic Regression
print("\n--- LASSO Logistic Regression ---")
X_lasso_train, X_lasso_test, y_lasso_train, y_lasso_test = train_test_split(X_lasso, y, test_size=0.1, random_state=42)

lasso_model = LogisticRegressionCV(
    penalty='l1',
    solver='liblinear',
    cv=5,
    random_state=42,
    max_iter=1000
)
lasso_model.fit(X_lasso_train, y_lasso_train)

y_pred_lasso = lasso_model.predict(X_lasso_test)

lasso_accuracy = accuracy_score(y_test, y_pred_lasso)
print("\nAccuracy (LASSO):", lasso_accuracy)


model_results.append({
    "Model": "LASSO Logistic Regression",
    "Accuracy": lasso_accuracy
})

## 2. Logistic Regression
print("\n--- Logistic Regression ---")
logit_model = LogisticRegression(max_iter=500000, random_state=42)
logit_model.fit(X_train, y_train)


y_pred_logit = logit_model.predict(X_test)


logit_accuracy = accuracy_score(y_test, y_pred_logit)
print("\nAccuracy (Logistic Regression):", logit_accuracy)


model_results.append({
    "Model": "Logistic Regression",
    "Accuracy": logit_accuracy
})

## 3. XGBoost Logistic Regression
print("\n--- XGBoost Logistic Regression ---")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Logistic regression objective
    eval_metric='logloss',        # Evaluation metric for binary classification
    random_state=42
)

# Fit the model on training data
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],     # Validation set for monitoring
    verbose=False                  # Print progress
)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("\nAccuracy (XGB Logistic Regression):", xgb_accuracy)

model_results.append({
    "Model": "XGBoost Logistic Regression",
    "Accuracy": xgb_accuracy
})

results_df = pd.DataFrame(model_results)