import argparse

from train import train_model
from evaluate import evaluate_model
from utils import *

def get_processor():
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    else:
        return "CPU"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TCN model with MLflow logging')
    parser.add_argument('--run_name', type=str, required=True, help='Give you run a name')
    parser.add_argument('--num_channels', type=int, default=64, help='Number of channels in TCN')
    parser.add_argument('--kernel_size', type=int, default=2, help='Kernel size for convolutions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file')        

    args = parser.parse_args()

    #platform name
    platform_name = "TCN"

    # Feature engineering
    data = perform_feature_engineering(file_path='Dataset/superstore-orders.csv')
    # Save to CSV after feature engineering
    data.to_csv('Dataset/engineered_superstore_orders.csv', index=False)

    # Prepare data
    X_test, X_train, y_test, y_train = prepare_data(file_path='Dataset/engineered_superstore_orders.csv')

    # Model training
    model, algo_params = train_model(X_train=X_train, y_train=y_train, num_channels=args.num_channels, kernel_size=args.kernel_size, dropout=args.dropout, learning_rate=args.learning_rate, num_epochs=args.num_epochs)

    # Save run ID and model
    run_id = mlflow.active_run().info.run_id
    save_run_id(run_id, experiment_name=args.run_name)
    print('Run ID is successfully saved!')
    save_trained_model(model, experiment_name=args.run_name)
    print('Model has been successfully saved!')

    experiment_name=args.run_name

    metrics, performance_score = evaluate_model(experiment_name, args.model_path, X_test, X_train, y_test, args.num_channels, args.kernel_size, args.dropout)
    print('successful evalaution!')

    extract_features_and_save_to_json("Dataset/engineered_superstore_orders.csv", 
                                      f"results/output_{experiment_name}.json", 
                                      algo_params, 
                                      algo_name="Sales Prediction Algorithm", 
                                      metrics=metrics, 
                                      processor=get_processor(), 
                                      platform_name=platform_name,
                                      performance_score=performance_score)