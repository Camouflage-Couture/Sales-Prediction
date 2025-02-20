import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.stats as stats
import mlflow
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def perform_feature_engineering(file_path):

    data = pd.read_csv(file_path)

    # Define the target variable
    data['Sales_per_Customer'] = data['Sales per Customer']
    # Add lag Features for 'Sales per Customer' column
    for lag in range(1, 11):  # Sales per Customer data from 1 to 10 rows ago
        data[f'sales_per_customer_lag_{lag}'] = data['Sales per Customer'].shift(lag)

    # Add rolling statistics features for 'Sales per Customer' column
    data['sales_per_customer_rolling_mean_7'] = data['Sales per Customer'].rolling(window=7).mean()
    data['sales_per_customer_rolling_std_7'] = data['Sales per Customer'].rolling(window=7).std()
    data['sales_per_customer_rolling_mean_30'] = data['Sales per Customer'].rolling(window=30).mean()
    data['sales_per_customer_rolling_std_30'] = data['Sales per Customer'].rolling(window=30).std()

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['Category', 'Sub-Category'], drop_first=True)

    # Handle missing values created by lag/rolling features
    data.fillna(method='bfill', inplace=True)

    # Drop original Sales per Customer column to avoid redundancy
    data.drop(columns=['Sales per Customer'], inplace=True)

    return data

def prepare_data(file_path):
    
    data = pd.read_csv(file_path)

    X = data.drop(columns=['Sales_per_Customer'])
    y = data['Sales_per_Customer'] # target

    # Ensure all columns are numeric
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
    y = pd.Series(y).apply(pd.to_numeric, errors='coerce').values

    # Handle missing values if any
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # visualizing distributions of target variables and features
    sns.histplot(y, kde=True)
    plt.show()

    # correlation matrix
    columns = ['Profit per Order','Quantity','Sales','Sales_per_Customer']
    corr_matrix = data[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # box plot for outliers
    data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), sharex=False, sharey=False, figsize=(15, 5))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #convert data into tensor format
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # channel dimension for conv1d
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    return X_test, X_train, y_test, y_train

def save_metrics_to_json(metrics, experiment_name):

    # save to json
    metrics_file_path = f'results/metrics/{experiment_name}_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f)

def extract_features_and_save_to_json(data_path, json_file_path, algorithm_params, algo_name, metrics, processor, platform_name, performance_score):

    data = pd.read_csv(data_path)
    
    # Extracting feature names from the DataFrame
    feature_names = list(data.columns)
    
    # Extracting all values for each feature
    feature_values = data.to_dict(orient='list')
    
    # Preparing the data for JSON output
    result_json = {
        "feature_rank_headings": feature_names,
        "feature_rank_data": feature_values,
        "alg_param": algorithm_params,
        "alg_name": algo_name,  # Update it as per your algorithm
        "evaluation_metric": metrics,
        "processor": processor,
        "platform": platform_name,
        "performance": performance_score,
        #"predicted_class": "Won",
        #"image_roi": "",
        #"classes": ["Won", "Loss"],
        #"customers": ["Cardio Ltd.", "Ebster Enterprise", "Sigmentia Inc.", "Countera Ltd.", "Oxland Gym Inc.", "Dilimator Labs Inc.", "Bell Voice Inc.", "Greenbeds Ltd.", "Tiobinos Technologies Ltd."],
        #"customers_ratings": ["97", "93", "91", "89", "55", "44", "43", "12", "8"],
        #"customer_predicted_classes": ["Won", "Won", "Won", "Won", "Loss", "Loss", "Loss", "Loss", "Loss"],
        #"customer_image_roi": [],
        #"customer_all_class_performance": []
    }
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    # Saving the JSON output to a file
    with open(json_file_path, 'w') as json_file:
        json.dump(result_json, json_file, indent=4)

    return "Output saved successfully!"

def save_trained_model(model, experiment_name):

    # save trained model
    model_file_path = f'models/{experiment_name}_model.pth'
    torch.save(model.state_dict(), model_file_path)

def save_run_id(run_id, experiment_name):

    # save run id
    run_id_file_path = f'results/run_id/{experiment_name}_run_id.txt'
    with open(run_id_file_path, 'w') as f:
        f.write(run_id)

def load_run_id(experiment_name):
    run_id_file_path = f'results/run_id/{experiment_name}_run_id.txt'
    with open(run_id_file_path, 'r') as f:
        run_id = f.read().strip()
    return run_id

def plot_and_log_artifacts(y_test, predictions, residuals, experiment_name):
    # Plot actual vs predicted values with different colors
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y_test.numpy(), label='Actual', color='blue', alpha=0.6, marker='o', linestyle='None')
    plt.plot(predictions.numpy(), label='Predicted', color='orange', alpha=0.6, marker='x', linestyle='None')
    plt.legend()
    plt.title('Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Sales')
    plt.grid(True)
    actual_vs_predicted_path = f'results/images/{experiment_name}_actual_vs_predicted.png'
    plt.savefig(actual_vs_predicted_path)
    mlflow.log_artifact(actual_vs_predicted_path)

    # Residual plot
    plt.subplot(1, 2, 2)
    plt.scatter(predictions.numpy(), residuals, alpha=0.5, color='purple')
    plt.hlines(y=0, xmin=min(predictions.numpy()), xmax=max(predictions.numpy()), colors='r')
    plt.title('Residuals Plot')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.grid(True)
    residuals_plot_path = f'results/images/{experiment_name}_residuals_plot.png'
    plt.savefig(residuals_plot_path)
    mlflow.log_artifact(residuals_plot_path)

    plt.tight_layout()
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color='green')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    residuals_distribution_path = f'results/images/{experiment_name}_residuals_distribution.png'
    plt.savefig(residuals_distribution_path)
    mlflow.log_artifact(residuals_distribution_path)
    plt.show()

    # Scatter plot of actual vs predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test.numpy(), predictions.numpy(), alpha=0.5)
    plt.plot([min(y_test.numpy()), max(y_test.numpy())], [min(y_test.numpy()), max(y_test.numpy())], color='red', linestyle='--')
    plt.title('Actual vs Predicted Scatter Plot')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    actual_vs_predicted_scatter_path = f'results/images/{experiment_name}_actual_vs_predicted_scatter.png'
    plt.savefig(actual_vs_predicted_scatter_path)
    mlflow.log_artifact(actual_vs_predicted_scatter_path)
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(7, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    qq_plot_path = f'results/images/{experiment_name}_qq_plot.png'
    plt.savefig(qq_plot_path)
    mlflow.log_artifact(qq_plot_path)
    plt.show()



