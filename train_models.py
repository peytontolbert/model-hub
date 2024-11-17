import argparse
import os
from pipeline.trainer import ModelTrainer
from pipeline.model_generator import ModelGenerator
from pipeline.dataset_manager import get_dataset_loaders
import json

def train_models(models_dir, results_dir, training_config):
    generator = ModelGenerator()
    trainer = ModelTrainer(training_config)
    os.makedirs(results_dir, exist_ok=True)

    for model_file in os.listdir(models_dir):
        if model_file.endswith('.json'):
            model_path = os.path.join(models_dir, model_file)
            try:
                with open(model_path, 'r') as f:
                    config = json.load(f)
                model = generator.generate_model(config)
                
                if model is None:
                    print(f"Error: Failed to generate model from {model_file}")
                    continue
                    
                train_loader, val_loader = get_dataset_loaders(training_config['dataset_name'], training_config['batch_size'])
                trainer.train_model(model, train_loader, val_loader)
                # Save results as needed
                print(f"Successfully trained model {model_file}")
            except Exception as e:
                print(f"Error processing {model_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train generated models')
    parser.add_argument('--models_dir', type=str, default='models/generated/', help='Directory with generated model configs')
    parser.add_argument('--results_dir', type=str, default='experiments/results/', help='Directory to save results')
    parser.add_argument('--dataset_name', type=str, default='dataset1', help='Dataset to use for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train each model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--task_type', type=str, default='classification', help='Task type: classification or regression')
    args = parser.parse_args()

    training_config = {
        'dataset_name': args.dataset_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'task_type': args.task_type
    }

    train_models(args.models_dir, args.results_dir, training_config)
