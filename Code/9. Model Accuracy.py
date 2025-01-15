# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:56:37 2025

@author: Yannick
"""

roc_data = []

# XP Difference Logistic Regression
xpdiff_probs = xpdiff_model.predict_proba(X_test[['blueExperienceDiff']])[:, 1]
xpdiff_auc = roc_auc_score(y_test, xpdiff_probs)
fpr_xpdiff, tpr_xpdiff, _ = roc_curve(y_test, xpdiff_probs)
roc_data.append(("ExperienceDiff Logistic Regression", fpr_xpdiff, tpr_xpdiff, xpdiff_auc))

# Dragon Logistic Regression
dragon_probs = dragon_model.predict_proba(X_test[['blueDragons']])[:, 1]
dragon_auc = roc_auc_score(y_test, dragon_probs)
fpr_dragon, tpr_dragon, _ = roc_curve(y_test, dragon_probs)
roc_data.append(("Dragon Logistic Regression", fpr_dragon, tpr_dragon, dragon_auc))

# Logistic Regression
logit_probs = logit_model.predict_proba(X_test)[:, 1]
logit_auc = roc_auc_score(y_test, logit_probs)
fpr_logit, tpr_logit, _ = roc_curve(y_test, logit_probs)
roc_data.append(("Logistic Regression", fpr_logit, tpr_logit, logit_auc))

# LASSO Logistic Regression
lasso_probs = lasso_model.predict_proba(X_lasso_test)[:, 1]
lasso_auc = roc_auc_score(y_lasso_test, lasso_probs)
fpr_lasso, tpr_lasso, _ = roc_curve(y_lasso_test, lasso_probs)
roc_data.append(("LASSO Logistic Regression", fpr_lasso, tpr_lasso, lasso_auc))

# XGB Logistic Regression
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
roc_data.append(("XGBoost Logistic Regression", fpr_xgb, tpr_xgb, xgb_auc))

# Random Forest
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_data.append(("Random Forest", fpr_rf, tpr_rf, rf_auc))

# k-Nearest Neighbors
knn_probs = knn_best.predict_proba(X_test)[:, 1]
knn_auc = roc_auc_score(y_test, knn_probs)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
roc_data.append(("k-Nearest Neighbors", fpr_knn, tpr_knn, knn_auc))


plt.figure(figsize=(10, 7))
for model_name, fpr, tpr, auc in roc_data:
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")


plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")


plt.title("ROC Curves for All Models", fontsize=16)
plt.xlabel("False Positive Rate (FPR)", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()