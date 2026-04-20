import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns

url = "data/diabetic_data.csv"
df = pd.read_csv(url)

# print(df.head(5))

# Drop or fill missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)  

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('readmitted', columns=1)  # defining
y = df['readmitted']
