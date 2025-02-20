import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.stats as stats
#import mlflow

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from autoPyTorch.api.tabular_regression import TabularRegressionTask
# from autoPyTorch.pipeline.components.setup.network_backbone.utils import SearchSpaceUpdates
# from ConfigSpace.hyperparameters import CategoricalHyperparameter as CSH


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

    for column in data.columns:
        # Attempt to convert each column to numeric, force non-convertible values to NaN
        data[column] = pd.to_numeric(data[column], errors='coerce')
    # Fill NaNs with a defined strategy, here zero is used
    data.fillna(0, inplace=True)  # Handling NaNs by replacing them with 0

    X = data.drop(columns=['Sales_per_Customer'])
    y = data['Sales_per_Customer']  # target

    X = np.nan_to_num(X)  # Further handling any slipped NaNs
    y = np.nan_to_num(y)

    sns.histplot(y, kde=True)
    plt.show()

    columns = ['Profit per Order', 'Quantity', 'Sales', 'Sales_per_Customer']
    corr_matrix = data[columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

    data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), sharex=False, sharey=False, figsize=(15, 5))
    plt.show()

    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_test, X_train, y_test, y_train

# def prepare_data(file_path):
    
#     data = pd.read_csv(file_path)

#     X = data.drop(columns=['Sales_per_Customer'])
#     y = data['Sales_per_Customer'] # target

#     # Ensure all columns are numeric
#     X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
#     y = pd.Series(y).apply(pd.to_numeric, errors='coerce').values

#     # Handle missing values if any
#     X = np.nan_to_num(X, nan=0.0)
#     y = np.nan_to_num(y, nan=0.0)

#     # visualizing distributions of target variables and features
#     sns.histplot(y, kde=True)
#     plt.show()

#     # correlation matrix
#     columns = ['Profit per Order','Quantity','Sales','Sales_per_Customer']
#     corr_matrix = data[columns].corr()
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#     plt.show()

#     # box plot for outliers
#     data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), sharex=False, sharey=False, figsize=(15, 5))
#     plt.show()

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # #convert data into tensor format
#     # X_train = torch.tensor(X_train, dtype=torch.float32)
#     # X_test = torch.tensor(X_test, dtype=torch.float32)
#     # y_train = torch.tensor(y_train, dtype=torch.float32)
#     # y_test = torch.tensor(y_test, dtype=torch.float32)

#     # # channel dimension for conv1d
#     # X_train = X_train.unsqueeze(1)
#     # X_test = X_test.unsqueeze(1)

#     return X_test, X_train, y_test, y_train

from autoPyTorch.pipeline.components.setup.network_head.base_network_head import NetworkHeadComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

def get_pytorch_regression_head_search_space():
    search_space = HyperparameterSearchSpace(
        hyperparameter="head_type",  # Positional argument 1: name
        value_range=["fully_connected"],  # Positional argument 2: list of choices
        default_value="fully_connected",  # Positional argument 3: default
    )
    
    # Directly access the ConfigurationSpace of the search space
    cs = search_space.configuration_space  
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "head_type",
            NetworkHeadComponent.get_available_components(),
            default_value="fully_connected",
        )
    )
    
    return search_space

X_test, X_train, y_test, y_train = prepare_data(file_path='dataset/engineered_superstore_orders.csv')

data = perform_feature_engineering('dataset/superstore-orders.csv')
data.to_csv('dataset/engineered_superstore_orders.csv', index=False)

# from autoPyTorch.pipeline.components.setup.network_backbone.utils import SearchSpaceUpdates

# X_test, X_train, y_test, y_train = prepare_data(file_path='dataset/engineered_superstore_orders.csv')
api = TabularRegressionTask(network_head_search_space=get_pytorch_regression_head_search_space()) 
 # use this custom search space
api.search(
            X_test=X_test,
            X_train=X_train,
            y_test=y_test,
            y_train=y_train,
            optimize_metric='r2',
            total_walltime_limit=1200,
            func_eval_time_limit_secs=200
        )

y_pred = api.predict(X_test)
y_pred = np.array(y_pred).reshape(-1)

mse = nn.MSELoss()(y_test, y_pred).item()
mae = nn.L1Loss()(y_test, y_pred).item()
rmse = torch.sqrt(nn.MSELoss()(y_test, y_pred)).item()
r2 = 1 - mse / torch.var(y_test).item()

print(f'MSE = {mse}')
print(f'MAE = {mae}')
print(f'RMSE = {rmse}')
print(f'R2 Score = {r2}')