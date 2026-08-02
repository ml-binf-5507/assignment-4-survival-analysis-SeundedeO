import pandas as pd

# Load the dataset
df = pd.read_excel("data/RADCURE-clinical-data.xlsx")

# Show the first few rows
print(df.head())

# Show all column names
print("\nColumns:")
print(df.columns.tolist())
