# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:56:01 2025

@author: Yannick
"""

feature_importance_data = []

# --- LINEAR MODELS ---

## 1. LASSO Logistic Regression
lasso_importances = pd.DataFrame({
    'Feature': X_lasso.columns,
    'Importance': np.abs(lasso_model.coef_[0])
}).sort_values(by='Importance', ascending=False)

feature_importance_data.append({
    'Model': 'LASSO Logistic Regression',
    'Data': lasso_importances
})

## 2. Logistic Regression
logit_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(logit_model.coef_[0])
}).sort_values(by='Importance', ascending=False)

feature_importance_data.append({
    'Model': 'Logistic Regression',
    'Data': logit_importances
})

## 3. XGB Logistic Regression
xgb_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance_data.append({
    'Model': 'XGBoost Logistic Regression',
    'Data': xgb_importances
})

# --- NON-LINEAR MODELS ---

## 1. Random Forest
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance_data.append({
    'Model': 'Random Forest',
    'Data': rf_importances
})

# --- PLOTTING VARIABLE IMPORTANCE ---


for model_data in feature_importance_data:
    model_name = model_data['Model']
    data = model_data['Data']


    plt.figure(figsize=(10, 6))
    plt.barh(data['Feature'], data['Importance'], color='skyblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.title(f'Variable Importance - {model_name}', fontsize=14)
    plt.gca().invert_yaxis()  # Inverted y axis for better readability
    plt.tight_layout()
    plt.show()
