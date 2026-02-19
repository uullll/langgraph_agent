import pandas as pd

# Load dataset
try:
    df = pd.read_parquet('dataset.parquet')
    print("\nDataset loaded successfully")
    print("\nFirst 5 rows:")
    print(df.head())

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Basic statistics
    print("\nKey statistics:")
    print(df.describe())

    # Data types
    print("\nColumn data types:")
    print(df.dtypes)

    # Save results
    df.describe().to_csv('initial_analysis_results.csv', index=False)
    with open('initial_analysis_summary.txt', 'w') as f:
        f.write("Dataset Analysis Summary\n")
        f.write("\nRows: " + str(len(df)) + "\n")
        f.write("Columns: " + str(len(df.columns)) + "\n")
        f.write("Missing values: " + str(df.isnull().sum().sum()) + "\n")

except Exception as e:
    print("Error processing dataset:", str(e))