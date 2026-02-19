import pandas as pd

# Load dataset from parquet file
df = pd.read_parquet('dataset.parquet')

# Display basic information about the dataframe
print('DataFrame shape:', df.shape)
print('\nFirst 5 rows:\n', df.head())
print('\nData types:\n', df.dtypes)
print('\nSummary statistics:\n', df.describe())