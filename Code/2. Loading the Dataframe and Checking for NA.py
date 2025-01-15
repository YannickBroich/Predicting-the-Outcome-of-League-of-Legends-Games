# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:50:18 2025

@author: Yannick
"""

url = "https://drive.google.com/uc?id=1st-b0Y-nlhQ3ob42pm3qqZF8fLSTBraO&export=download"

LolDataframe = pd.read_csv(url)

missing_values = LolDataframe.isnull().sum()
print("Missing Values per Column:")
print(missing_values)