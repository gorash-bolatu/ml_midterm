import pandas as pd

# Load the dataset
df = pd.read_csv('PROCESSED.csv')
print(df.info())

# Feature scaling for numerical columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols].head())

df_out = df
filename = 'PROCESSED_SCALED.csv'
df_out.to_csv(filename, index=False)
print("Wrote", df_out.shape[0], "rows and", df_out.shape[1], "columns to", filename)
