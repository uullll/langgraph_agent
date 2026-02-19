import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_parquet('dataset.parquet')
print('Dataset loaded successfully')
print(f'First 2 rows: \n{df.head(2)}')

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f'Numerical columns found: {numerical_cols}')

# Create histograms for top 10 numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols[:10], 1):
    plt.subplot(5, 2, i)
    df[col].hist(bins=20)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('top10_histograms.png')
print('Top 10 histograms saved')

# Box plot for absences
plt.figure(figsize=(8, 6))
df['absences'].plot(kind='box')
plt.title('Absences Distribution')
plt.ylabel('Days')
plt.savefig('absences_boxplot.png')
print('Absences box plot saved')

# Scatter plot matrix for grade correlations
pd.plotting.scatter_matrix(df[['G1', 'G2', 'G3'] + numerical_cols[:4]],
                          figsize=(12, 12),
                          diagonal='hist')
plt.suptitle('Grade Correlation Analysis')
plt.savefig('grade_correlation_scatter.png')
print('Grade correlation scatter matrix saved')