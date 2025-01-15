# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:51:23 2025

@author: Yannick
"""

LolDataframe = LolDataframe.drop(columns=[col for col in LolDataframe.columns if "red" in col.lower() or col == "gameId"])




correlation_matrix = LolDataframe.corr()

plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlationmatrix Blue on Blue")
plt.show()

numeric_features = LolDataframe.drop(columns=["blueWins"])


vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(numeric_features.shape[1])]

print(vif_data)