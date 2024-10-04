
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('C:/Users/user/Desktop/Github/BatiBank_CreditScoring/data/data.csv')

# Overview of the data
def data_overview(data):
    print(data.info())
    #print(data.describe())
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
    # For example, we could define outliers as anything below -10000 or above 100000
    filtered_data = data[(data[column] >=-25000 ) & (data[column] <= 25000)]
    
    plt.figure(figsize=(10, 6))
    
    # Create the histogram with KDE
    sns.histplot(filtered_data[column], bins=50, kde=True)  # KDE adds a smooth curve
    
    # Set x and y limits for better visibility
    plt.xlim(filtered_data[column].min() - 1000, filtered_data[column].max() + 1000)
    plt.ylim(0, filtered_data[column].count() // 10)  # Set y limit based on the count divided by 10 for better spacing
    
    plt.title(f'Distribution of {column} (filtered to show values between -25000 and 25000)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)  # Adding grid for better readability
    plt.axvline(0, color='red', linestyle='--')  # Add a vertical line at 0 for reference
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

# Outlier Detection (Vertical)
def boxplot_outliers(data, column):
    if column not in data.columns:
        print(f"Column '{column}' not found in the DataFrame.")
        return
    
    plt.figure(figsize=(10, 6))

    # Create a vertical boxplot
    sns.boxplot(y=data[column])

    # Title and labels with larger font sizes
    plt.title(f'Boxplot of {column}', fontsize=16)
    plt.ylabel(column, fontsize=14)  # Label for the y-axis
    plt.xlabel('Value', fontsize=14)  # Label for the x-axis

    # Set y-ticks explicitly based on the range of data
    y_ticks = range(int(data[column].min()), int(data[column].max()) + 1, 1000)  # Customize step as needed
    plt.yticks(ticks=y_ticks, fontsize=12)

    plt.grid(True)  # Add grid for better visibility
    plt.xticks([])  # Remove x-ticks to focus on y-axis
    plt.show()