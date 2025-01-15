# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:55:10 2025

@author: Yannick
"""

# This takes 5 minutes to run. Feel free to make a sandwich or use the restroom
# :)

##k-Nearest Neighbors
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

print("\n--- k-Nearest Neighbors (kNN) ---")


K = 50
val_acc = []
train_acc = []


for k in range(1, K + 1):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)


    train_acc.append(knn_model.score(X_train, y_train))
    val_acc.append(knn_model.score(X_val, y_val))


k_star = np.argmax(val_acc) + 1
print(f"Best model achieved validation accuracy of {max(val_acc):.4f} with k = {k_star}")


knn_best = KNeighborsClassifier(n_neighbors=k_star)
knn_best.fit(X_train, y_train)
test_accuracy_knn = knn_best.score(X_test, y_test)
print(f"Test accuracy (kNN): {test_accuracy_knn:.4f}")


model_results.append({
    "Model": "k-Nearest Neighbors",
    "Accuracy": test_accuracy_knn
})

## 2. Random Forest
print("\n--- Random Forest ---")

# Split the data into train and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

train_acc =np.full(X_train.shape[1], np.nan)
test_acc =np.full(X_train.shape[1], np.nan)

for max_features in range(1, X_train.shape[1] + 1):
    # fit the model
    rf = RandomForestClassifier(n_estimators=100, max_features=max_features, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    # predict
    y_pred_val = rf.predict(X_val)
    y_pred_train = rf.predict(X_train)
    # compute accuracy
    train_acc[max_features - 1] = np.mean(y_train == y_pred_train)
    test_acc[max_features - 1] = np.mean(y_val == y_pred_val)

m_star = np.argmax(test_acc) + 1  # Add 1 because Python indices start at 0
# print(f"Best max_features: {m_star}")

ntrees = np.arange(50, 1001, 50)
train_acc = np.full(len(ntrees), np.nan)
test_acc = np.full(len(ntrees), np.nan)

for n, ntrees_val in enumerate(ntrees):
    # Fit the model
    rf = RandomForestClassifier(n_estimators=ntrees_val, max_features=m_star, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    # Predict
    y_pred_train = rf.predict(X_train)
    y_pred_val = rf.predict(X_val)
    # Compute accuracies
    train_acc[n] = np.mean(y_train == y_pred_train)
    test_acc[n] = np.mean(y_val == y_pred_val)

rf_model = RandomForestClassifier(n_estimators=ntrees[m_star-1], max_features=m_star, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# rf_prob = rf_model.predict_proba(X_test)


feature_importances_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Random Forest):\n")
print(feature_importances_rf)


y_pred_rf = rf_model.predict(X_test)

#Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAccuracy (Random Forest): {accuracy_rf:.4f}")


model_results.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_rf
})


results_df = pd.DataFrame(model_results)

print(results_df)