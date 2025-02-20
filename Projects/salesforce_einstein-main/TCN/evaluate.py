import torchmetrics
import mlflow
import argparse

from utils import *
from model import TCN

def evaluate_model(experiment_name, model_path, X_test, X_train, y_test, num_channels, kernel_size, dropout):

    # prepare data for evaluation
    # X_test, X_train, y_test, y_train = prepare_data(data_path)

    # load the saved model
    model = TCN(num_inputs=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, X_train=X_train)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        predictions = model(X_test).squeeze()

    # Calculate metrics
        # Initialize the metrics
        mse_metric = torchmetrics.MeanSquaredError()
        mae_metric = torchmetrics.MeanAbsoluteError()
        rmse_metric = torchmetrics.MeanSquaredError(squared=False)  # RMSE is the square root of MSE
        r2_metric = torchmetrics.R2Score()
        evs_metric = torchmetrics.ExplainedVariance()

        # Compute metrics
        mse = mse_metric(predictions, y_test).item()
        mae = mae_metric(predictions, y_test).item()
        rmse = rmse_metric(predictions, y_test).item()
        r2 = r2_metric(predictions, y_test).item()
        evs = evs_metric(predictions, y_test).item()

        max_rmse = 200
        performance_score = rmse_to_score(rmse, max_rmse)

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Variance': evs 
        }

        #save_metrics_to_json(metrics, experiment_name)

        # Log metrics with MLflow
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('Explained_Variance_Score', evs)
        mlflow.log_metric('Performance Score', performance_score)

        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'R-squared: {r2:.4f}')
        print(f'Explained Variance Score: {evs:.4f}')
        print(f"Performance Score: {performance_score:.2f} out of 100")

    # Call the plotting function
    residuals = y_test.numpy() - predictions.numpy()
    plot_and_log_artifacts(y_test, predictions, residuals, experiment_name)

    return metrics, performance_score

def rmse_to_score(rmse, max_rmse):
    if rmse > max_rmse:
        return 0  # Prevent negative scores
    normalized_rmse = rmse / max_rmse
    score = (1 - normalized_rmse) * 100
    return max(0, score)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Evaluate TCN model and save metrics')
#     parser.add_argument('--run_name', type=str, required=True, help='Give you run a name')
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')
#     parser.add_argument('--data_path', type=str, required=True, help='Path to the input CSV file for evaluation')
#     parser.add_argument('--num_channels', type=int, default=64, help='Number of channels in TCN')
#     parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size for convolutions')
#     parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
#     parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimizer')
#     args = parser.parse_args()

#     experiment_name = args.run_name
#     run_id = load_run_id(experiment_name)

#     with mlflow.start_run(run_id=run_id):
#         evaluate_model(args.model_path, args.data_path, args.num_channels, args.kernel_size, args.dropout)
#     mlflow.end_run()



    