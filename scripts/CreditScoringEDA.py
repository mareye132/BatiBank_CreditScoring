import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('C:/Users/user/Desktop/Github/BatiBank_CreditScoring/data/data.csv')

# Overview of the data
def data_overview(data):
    print(data.info())
    #print(data.describe())  # You can uncomment to see statistical summary if needed
    #print(data.head())

# Summary statistics for numerical columns
def summary_statistics(data):
    print(data.describe())

# Distribution of numerical features
def plot_numerical_distribution(data, column):
    # Check if the column exists in the DataFrame
    if column not in data.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return
    
    # Filter out any extreme outliers if needed (optional)
    filtered_data = data[(data[column] >= -25000) & (data[column] <= 25000)]
    
    plt.figure(figsize=(10, 6))
    
    # Create the histogram with KDE
    sns.histplot(filtered_data[column], bins=50, kde=True)
    
    # Set x and y limits for better visibility
    plt.xlim(filtered_data[column].min() - 1000, filtered_data[column].max() + 1000)
    plt.ylim(0, filtered_data[column].count() // 10)
    
    plt.title(f'Distribution of {column} (filtered to show values between -25000 and 25000)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at 0 for reference
    plt.show()

# Distribution of categorical features
def plot_categorical_distribution(data, column):
    if column not in data.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x=data[column], palette="Set2")
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Correlation matrix
def correlation_matrix(data):
    # Select only the numerical columns for correlation calculation
    numerical_data = data.select_dtypes(include=['float64', 'int64'])

    if numerical_data.empty:
        print("No numerical columns available for correlation analysis.")
        return
    
    # Compute the correlation matrix
    corr_matrix = numerical_data.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Check for missing values
def missing_values(data):
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    print(f"Missing values percentage:\n{missing_percentage}")

# Outlier Detection (Vertical boxplot)
def boxplot_outliers(data, column):
    if column not in data.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[column])
    
    plt.title(f'Boxplot of {column}', fontsize=16)
    plt.ylabel(column, fontsize=14)
    plt.xlabel('Value', fontsize=14)
    
    y_ticks = range(int(data[column].min()), int(data[column].max()) + 1, 1000)
    plt.yticks(ticks=y_ticks, fontsize=12)
    
    plt.grid(True)
    plt.xticks([])
    plt.show()

# Analyze all numerical and categorical features in the dataset
def analyze_data(data):
    numerical_columns = ['Amount', 'Value']  # Add other numerical fields if needed
    categorical_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                           'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
                           'ChannelId', 'PricingStrategy', 'FraudResult']  # Categorical fields
    
    # Plot distribution of each numerical column
    print("Numerical Feature Distributions:")
    for col in numerical_columns:
        print(f"Plotting distribution for numerical feature: {col}")
        plot_numerical_distribution(data, col)
    
    # Plot distribution of each categorical column
    print("Categorical Feature Distributions:")
    for col in categorical_columns:
        print(f"Plotting distribution for categorical feature: {col}")
        plot_categorical_distribution(data, col)
    
    # Plot the correlation matrix for numerical features
    print("Correlation Matrix:")
    correlation_matrix(data)
