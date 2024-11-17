# Model Factory

A flexible framework for generating, training, and evaluating diverse neural network architectures. This project automatically generates and tests various neural network architectures, supporting multiple architectural patterns and combinations.

## Features

- **Diverse Architecture Generation**
  - Convolutional Neural Networks (CNN)
  - Multi-Layer Perceptrons (MLP)
  - Residual Networks (ResNet-style)
  - Dense Networks (DenseNet-style)
  - Transformer-based architectures
  - Hybrid architectures
  - Inception-style networks
  - Recurrent architectures (LSTM, GRU)
  - Attention-based models

- **Advanced Components**
  - Skip connections
  - Parallel paths
  - Multi-head attention
  - Various normalization techniques
  - Multiple activation functions
  - Regularization methods

- **Training and Evaluation**
  - Automated model training
  - Performance evaluation
  - Results analysis
  - Configurable hyperparameters

## Project Structure
```
model_factory/
├── pipeline/
│ ├── model_generator.py # Architecture generation logic
│ ├── trainer.py # Model training implementation
│ ├── dataset_manager.py # Dataset handling
│ └── utils.py # Utility functions
├── models/
│ └── generated/ # Generated model configurations
├── experiments/
│ └── results/ # Training results
├── scripts/
│ ├── generate_models.py # Model generation script
│ ├── train_models.py # Training script
│ └── analyze_results.py # Results analysis
└── README.md
```


## Installation
```bash
Clone the repository
git clone https://github.com/peytontolbert/model_factory.git
cd model_factory
Install dependencies
pip install -r requirements.txt
```


## Usage

### 1. Generate Models

Generate a set of unique model architectures:
```bash
python generate_models.py --num_models 100 --save_dir models/generated/
```

Options:
- `--num_models`: Number of models to generate
- `--save_dir`: Directory to save generated model configurations

### 2. Train Models

Train the generated models:
```bash
python train_models.py --models_dir models/generated/ --results_dir experiments/results/
```


Options:
- `--models_dir`: Directory containing model configurations
- `--results_dir`: Directory to save training results
- `--dataset_name`: Dataset to use (default: 'dataset1')
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 5)
- `--learning_rate`: Learning rate (default: 0.001)
- `--task_type`: Task type (classification/regression)

### 3. Analyze Results

Analyze training results:
```bash
python scripts/analyze_results.py --results_dir experiments/results/
```


## Supported Layer Types

- **Convolutional Layers**: Conv1d, Conv2d, Conv3d, ConvTranspose1d/2d/3d
- **Pooling Layers**: MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptivePool
- **Normalization**: BatchNorm1d/2d/3d, LayerNorm, InstanceNorm, GroupNorm
- **Activation Functions**: ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh
- **Attention Mechanisms**: MultiheadAttention, TransformerEncoder
- **Recurrent Layers**: LSTM, GRU, RNN
- **Other**: Linear, Dropout, Flatten, Embedding

## Custom Blocks

- **InceptionBlock**: Parallel path processing
- **ResidualBlock**: Skip connections
- **CustomTransformerBlock**: Self-attention with feed-forward
- **DenseBlock**: Dense connectivity pattern

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Various architecture papers that inspired the implementations