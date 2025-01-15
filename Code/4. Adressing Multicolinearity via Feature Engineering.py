# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:51:10 2025

@author: Yannick
"""

# Dropping redundant columns and creating new features
LolDataframe['blueKDA'] = (LolDataframe['blueKills'] + LolDataframe['blueAssists']) / LolDataframe['blueDeaths'].replace(0, 1)

LolDataframe['blueVisionScore'] = LolDataframe['blueWardsPlaced'] + LolDataframe['blueWardsDestroyed']

LolDataframe['blueTotalJungleUnits'] = LolDataframe['blueTotalJungleMinionsKilled'] + LolDataframe['blueDragons']


LolDataframe = LolDataframe.drop(columns=[
    'blueTotalGold',
    'blueGoldDiff',
    'blueTotalExperience',
    'blueKills',
    'blueDeaths',
    'blueAssists',
    'blueWardsPlaced',
    'blueWardsDestroyed',
    'blueEliteMonsters',
    'blueTotalJungleMinionsKilled',
    'blueTotalMinionsKilled',

])


# Select only numeric features for VIF calculation
numeric_features = LolDataframe.select_dtypes(include=['float64', 'int64'])

# Calculate VIF for each numeric feature
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_features.values, i) for i in range(numeric_features.shape[1])]

print(vif_data)