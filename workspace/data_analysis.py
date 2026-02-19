import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_parquet('dataset.parquet')

# Check missing values
print('Missing values:\n', df.isnull().sum())

# Distribution of grades (G1, G2, G3)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
df['G1'].hist(bins=10)
plt.title('G1 Distribution')

plt.subplot(1, 3, 2)
df['G2'].hist(bins=10)
plt.title('G2 Distribution')

plt.subplot(1, 3, 3)
df['G3'].hist(bins=10)
plt.title('G3 Distribution')
plt.tight_layout()
plt.savefig('grades_distribution.png')

# Absences analysis
plt.figure(figsize=(8, 5))
df['absences'].plot(kind='box')
plt.title('Absences Distribution (Boxplot)')
plt.savefig('absences_boxplot.png')

# Top 10 students by final grade (G3)
top_10_students = df.sort_values(by='G3', ascending=False).head(10)
print('\nTop 10 students by G3:\n', top_10_students[['school', 'sex', 'age', 'G3']])

# Save top 10 students to CSV
top_10_students.to_csv('top_10_students.csv', index=False)