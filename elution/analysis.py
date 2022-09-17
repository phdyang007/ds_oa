#data analysis, summary

import pandas as pd 
import numpy as np 


df_raw = pd.read_excel("data.xlsx")

print(df_raw.describe())
print(df_raw.info())

df_dropnan = df_raw.dropna()


print(df_dropnan.describe())
print(df_dropnan.info())

df_dropnan.to_excel("data_dropna.xlsx")