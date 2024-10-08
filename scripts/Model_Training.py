
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Split the data
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Model Training function
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    
    return trained_models

# Hyperparameter tuning using Grid Search
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Model Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

# Create a preprocessing and model pipeline
def create_pipeline(model):
    # Updated with the actual categorical and numeric columns in the dataset
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    numeric_cols = ['Amount', 'Value']  # Numeric columns to scale

    # Preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),  # Scale numeric features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-Hot Encode categorical features and ignore unknown categories
        ]
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Main function
if __name__ == "__main__":
    # Load the data
    df = load_data('C:/Users/user/Desktop/Github/BatiBank_CreditScoring/data/data.csv')

    # Drop unnecessary identifier columns (if any)
    df_cleaned = df.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'], errors='ignore')

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df_cleaned, 'FraudResult')

    # Create a pipeline for each model and train
    trained_models = {}
    for model_name, model in {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }.items():
        pipeline = create_pipeline(model)
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    # Tune Hyperparameters for Random Forest using the pipeline
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20]
    }
    best_rf = tune_hyperparameters(trained_models['Random Forest'], param_grid_rf, X_train, y_train)

    # Evaluate models
    for model_name, model in trained_models.items():
        print(f"Evaluation for {model_name}")
        metrics = evaluate_model(model, X_test, y_test)
        print(metrics)

    # Evaluate the best tuned model
    print("Evaluation for Tuned Random Forest")
    metrics = evaluate_model(best_rf, X_test, y_test)
    print(metrics)
