
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('C:/Users/user/Desktop/Github/BatiBank_CreditScoring/data/data.csv')

# Overview of the data
def data_overview(data):
    print(data.info())
    print(data.describe())
    print(data.head())

# Summary statistics for numerical columns
def summary_statistics(data):
    print(data.describe())

# Distribution of numerical features
def plot_numerical_distribution(data, column):
    data[column].hist(bins=50)
    plt.title(f'Distribution of {column}')
    plt.show()

# Distribution of categorical features
def plot_categorical_distribution(data, column):
    sns.countplot(data[column])
    plt.title(f'Distribution of {column}')
    plt.show()

# Correlation matrix
def correlation_matrix(data):
    # Select only the numerical columns for correlation calculation
    numerical_data = data.select_dtypes(include=['float64', 'int64'])

    # Check if there are any numerical columns available
    if numerical_data.empty:
        print("No numerical columns available for correlation analysis.")
        return
    
    # Compute the correlation matrix
    corr_matrix = numerical_data.corr()

    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


# Check for missing values
def missing_values(data):
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    print(missing_percentage)

# Outlier detection using boxplot
def boxplot_outliers(data, column):
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# Example Usage:
# data_overview(data)
# summary_statistics(data)
# plot_numerical_distribution(data, 'Amount')
# plot_categorical_distribution(data, 'ProductCategory')
# correlation_matrix(data)
# missing_values(data)
# boxplot_outliers(data, 'Amount')
