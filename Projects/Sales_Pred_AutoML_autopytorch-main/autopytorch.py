import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from autoPyTorch.api.tabular_regression import TabularRegressionTask

def perform_feature_engineering(file_path):
    data = pd.read_csv(file_path)
    data['Sales_per_Customer'] = data['Sales per Customer']
    for lag in range(1, 11):
        data[f'sales_per_customer_lag_{lag}'] = data['Sales per Customer'].shift(lag)
    data['sales_per_customer_rolling_mean_7'] = data['Sales per Customer'].rolling(window=7).mean()
    data['sales_per_customer_rolling_std_7'] = data['Sales per Customer'].rolling(window=7).std()
    data['sales_per_customer_rolling_mean_30'] = data['Sales per Customer'].rolling(window=30).mean()
    data['sales_per_customer_rolling_std_30'] = data['Sales per Customer'].rolling(window=30).std()
    data = pd.get_dummies(data, columns=['Category', 'Sub-Category'], drop_first=True)
    data.fillna(method='bfill', inplace=True)
    data.drop(columns=['Sales per Customer'], inplace=True)
    return data

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data.fillna(0, inplace=True)
    X = data.drop(columns=['Sales_per_Customer'])
    y = data['Sales_per_Customer']
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def visualize_data(data):
    sns.histplot(data, kde=True)
    plt.show()
    columns = ['Profit per Order', 'Quantity', 'Sales', 'Sales_per_Customer']
    corr_matrix = data[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), sharex=False, sharey=False, figsize=(15, 5))
    plt.show()

# Data Preparation
engineered_data = perform_feature_engineering('dataset/superstore-orders.csv')
engineered_data.to_csv('dataset/engineered_superstore_orders.csv', index=False)
X_train, X_test, y_train, y_test = prepare_data('dataset/engineered_superstore_orders.csv')

# Visualization
# visualize_data(engineered_data)

# Auto-PyTorch Regression Model
api = TabularRegressionTask()
api.search(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, optimize_metric='r2', total_walltime_limit=1200, func_eval_time_limit_secs=200)

# Model Evaluation
y_pred = api.predict(X_test)
y_pred = np.array(y_pred).reshape(-1)
mse = nn.MSELoss()(torch.tensor(y_test), torch.tensor(y_pred)).item()
mae = nn.L1Loss()(torch.tensor(y_test), torch.tensor(y_pred)).item()
rmse = torch.sqrt(nn.MSELoss()(torch.tensor(y_test), torch.tensor(y_pred))).item()
r2 = 1 - mse / torch.var(torch.tensor(y_test)).item()
print(f'MSE = {mse}, MAE = {mae}, RMSE = {rmse}, R2 Score = {r2}')
