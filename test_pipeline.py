import unittest
import torch
from pipeline.model_generator import generate_model
from pipeline.trainer import train_model, evaluate_model
from pipeline.utils import get_data_loaders

class TestPipeline(unittest.TestCase):

    def test_generate_model(self):
        config = {
            "layers": [
                {"type": "Linear", "in_features": 10, "out_features": 5},
                {"type": "ReLU"},
                {"type": "Linear", "in_features": 5, "out_features": 2}
            ]
        }
        model = generate_model(config)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(len(model), 3)

    def test_train_model(self):
        config = {
            "layers": [
                {"type": "Linear", "in_features": 784, "out_features": 10}
            ]
        }
        model = generate_model(config)
        train_loader, test_loader = get_data_loaders('MNIST', batch_size=32)
        result = train_model(model, train_loader, test_loader, {'epochs': 1, 'learning_rate': 0.001})
        self.assertIn('loss', result)
        self.assertIn('accuracy', result)
        self.assertGreaterEqual(result['accuracy'], 0)
        self.assertLessEqual(result['accuracy'], 100)

    def test_evaluate_model(self):
        config = {
            "layers": [
                {"type": "Linear", "in_features": 784, "out_features": 10}
            ]
        }
        model = generate_model(config)
        _, test_loader = get_data_loaders('MNIST', batch_size=32)
        accuracy = evaluate_model(model, test_loader)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)

if __name__ == '__main__':
    unittest.main()
