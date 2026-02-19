import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_parquet('dataset.parquet')

# Display basic information
print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nData Types:")
print(data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Top 10 numerical features analysis
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
print("\nTop 10 Numerical Features:")
print(numerical_features[:10])

# Visualize top 10 numerical features
data[numerical_features[:10]].hist(bins=20, figsize=(15, 10))
plt.suptitle('Top 10 Numerical Features Distribution')
plt.savefig('top10_histograms.png')
plt.close()

# Absences analysis
print("\nAbsences Analysis:")
print(data['absences'].describe())

# Visualize absences distribution
plt.figure(figsize=(10, 6))
data['absences'].plot(kind='box')
plt.title('Absences Distribution')
plt.savefig('absences_boxplot.png')
plt.close()

# Grade correlation analysis
print("\nGrade Correlation Analysis:")
print(data[['G1', 'G2', 'G3']].corr())

# Visualize grade correlations
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.pairplot(data[['G1', 'G2', 'G3']])
plt.suptitle('Grade Correlation Scatter Matrix')
plt.savefig('grade_correlation_scatter.png')
plt.close()