import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("Moviedata.csv", encoding="latin1")

df.head()
df.info()

df = df.dropna(subset=['Name', 'Year', 'Duration', 'Votes', 'Rating'])
print("After dropna:", df.shape)
df = df.drop_duplicates(subset=['Name', 'Year', 'Director'], keep='first')
print("After dropping duplicates:", df.shape)

# --- Clean Year ---
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)



