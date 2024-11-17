import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.dataset_manager import get_dataset_loaders
from pipeline.utils import ensure_dir

class ModelTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

    def train_model(self, model, train_loader, val_loader):
        """
        Train and evaluate a model.
        """
        model.to(self.device)
        criterion = self.get_criterion()
        optimizer = optim.Adam(model.parameters(), lr=self.config.get("learning_rate", 0.001))

        num_epochs = self.config.get("epochs", 10)
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_accuracy = self.evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    def evaluate_model(self, model, val_loader):
        """
        Evaluate the model on the validation set.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def get_criterion(self):
        """
        Get the appropriate loss function based on the task.
        """
        task_type = self.config.get('task_type', 'classification')
        if task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif task_type == 'regression':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

if __name__ == "__main__":
    # Example usage
    from pipeline.model_generator import ModelGenerator

    # Load configuration
    training_config = {
        'dataset_name': 'dataset1',
        'batch_size': 64,
        'epochs': 5,
        'learning_rate': 0.001,
        'task_type': 'classification'
    }

    # Prepare data loaders
    train_loader, val_loader = get_dataset_loaders(training_config['dataset_name'], training_config['batch_size'])

    # Generate a model
    generator = ModelGenerator()
    config, _ = generator.generate_unique_model_config()
    model = generator.generate_model(config)

    # Train the model
    trainer = ModelTrainer(training_config)
    trainer.train_model(model, train_loader, val_loader)
