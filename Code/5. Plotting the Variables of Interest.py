# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:53:45 2025

@author: Yannick
"""

features = ['blueCSPerMin', 'blueGoldPerMin', 'blueKDA', 'blueVisionScore', 'blueTotalJungleUnits', 'blueHeralds']

# Generate boxplots for each feature
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='blueWins', y=feature, data=LolDataframe)
    plt.title(f'{feature} Distribution by Game Outcome', fontsize=14)
    plt.xlabel('Game Outcome (blueWins)', fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.show()
    

sns.kdeplot(data=LolDataframe[LolDataframe['blueWins'] == 0], x='blueGoldPerMin', label='Loss', fill=True)
sns.kdeplot(data=LolDataframe[LolDataframe['blueWins'] == 1], x='blueGoldPerMin', label='Win', fill=True)
plt.title('Density Plot of blueGoldPerMin by Game Outcome')
plt.xlabel('Gold Per Minute')
plt.ylabel('Density')
plt.legend()
plt.show()


sns.scatterplot(x='blueGoldPerMin', y='blueExperienceDiff', hue='blueWins', data=LolDataframe)
plt.title('Scatter Plot of Gold Per Minute vs Experience Difference')
plt.xlabel('Gold Per Minute')
plt.ylabel('Experience Difference')
plt.legend(title='Game Outcome')
plt.show()