import torch.nn as nn
import torch.optim as optim
import mlflow
import argparse

from utils import *
from model import TCN

def train_model(X_train, y_train, num_channels, kernel_size, dropout, learning_rate, num_epochs):

    # define model
    model = TCN(num_inputs=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, X_train=X_train)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,              # Learning rate
        betas=(0.9, 0.999),   # Coefficients for running averages
        eps=1e-8,             # Small constant for numerical stability
        weight_decay=1e-5,    # Weight decay (L2 penalty)
        amsgrad=True          # Use AMSGrad variant
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            mlflow.log_metric('loss', loss.item(), step=epoch+1)

    # Dictionary of algorithm parameters
    algo_params = {
        "num_channels": num_channels,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs
    }

    # MLFlow log_param
    mlflow.log_param("num_channels", num_channels)
    mlflow.log_param("kernel_size", kernel_size)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("learning_rate", 0.1)

    # saving the model
    mlflow.pytorch.log_model(model,'model')

    return model, algo_params

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='Train TCN model with MLflow logging')
#     parser.add_argument('--run_name', type=str, required=True, help='Give you run a name')
#     parser.add_argument('--num_channels', type=int, default=64, help='Number of channels in TCN')
#     parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size for convolutions')
#     parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
#     parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimizer')
#     parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
#     args = parser.parse_args()

#     # Feature engineering
#     data = perform_feature_engineering(file_path='Dataset/superstore-orders.csv')
#     # Save to CSV after feature engineering
#     data.to_csv('Dataset/engineered_superstore_orders.csv', index=False)

#     # Prepare data
#     X_test, X_train, y_test, y_train = prepare_data(file_path='Dataset/engineered_superstore_orders.csv')

#     # Model training
#     model = train_model(X_train=X_train, y_train=y_train, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout, learning_rate=args.learning_rate, num_epochs=args.num_epochs)

#     # Save run ID and model
#     run_id = mlflow.active_run().info.run_id
#     save_run_id(run_id, experiment_name=args.run_name)
#     print('Run ID is successfully saved!')
#     save_trained_model(model, experiment_name=args.run_name)
#     print('Model has been successfully saved!')


