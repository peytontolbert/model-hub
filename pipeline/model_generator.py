import json
import torch.nn as nn
import random
import hashlib
import itertools
from pipeline.utils import get_available_layers
import os
import torch
import torch.nn.functional as F

class ModelGenerator:
    def __init__(self):
        self.available_layers = get_available_layers()
        self.generated_hashes = set()

    def generate_unique_model_config(self):
        """
        Generate a unique model configuration.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            config = self.create_random_config()
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode()).hexdigest()

            if config_hash not in self.generated_hashes:
                self.generated_hashes.add(config_hash)
                return config, config_hash
        raise RuntimeError("Failed to generate a unique model configuration.")

    def create_random_config(self):
        """
        Create a random model configuration with diverse architectures including advanced patterns.
        """
        architecture_type = random.choice([
            'cnn', 'mlp', 'residual_net', 'dense_net', 
            'transformer', 'hybrid', 'inception_style',
            'recurrent', 'attention_based'
        ])
        
        layers = []
        current_dim = 784  # MNIST input size (28*28)
        current_channels = 1  # MNIST starts with 1 channel
        skip_connections = []
        
        if architecture_type == 'cnn':
            layers.extend(self._create_cnn_backbone(current_dim, current_channels))
        elif architecture_type == 'mlp':
            layers.extend(self._create_mlp_backbone(current_dim))
        elif architecture_type == 'residual_net':
            layers.extend(self._create_residual_backbone(current_dim, current_channels))
        elif architecture_type == 'dense_net':
            layers.extend(self._create_dense_backbone(current_dim, current_channels))
        elif architecture_type == 'transformer':
            layers.extend(self._create_transformer_backbone(current_dim))
        elif architecture_type == 'hybrid':
            layers.extend(self._create_hybrid_backbone(current_dim, current_channels))
        elif architecture_type == 'inception_style':
            layers.extend(self._create_inception_backbone(current_dim, current_channels))
        elif architecture_type == 'recurrent':
            layers.extend(self._create_recurrent_backbone(current_dim))
        elif architecture_type == 'attention_based':
            layers.extend(self._create_attention_backbone(current_dim))

        # Ensure output layer for MNIST (10 classes)
        layers.append({
            'type': 'Linear',
            'params': {
                'in_features': self._get_final_dim(layers),
                'out_features': 10
            }
        })

        config = {
            'layers': layers,
            'skip_connections': skip_connections,
            'architecture_type': architecture_type
        }
        return config

    def _create_cnn_backbone(self, input_dim, in_channels):
        """Create CNN-based architecture."""
        layers = []
        current_dim = input_dim
        current_channels = in_channels
        
        num_conv_blocks = random.randint(2, 6)
        for _ in range(num_conv_blocks):
            # Conv block with optional components
            block = []
            
            # 1. Convolution
            out_channels = random.randint(32, 256)
            kernel_size = random.choice([3, 5, 7])
            stride = random.choice([1, 2])
            padding = kernel_size // 2
            
            block.append({
                'type': 'Conv2d',
                'params': {
                    'in_channels': current_channels,
                    'out_channels': out_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding
                }
            })
            
            # 2. Normalization (optional)
            if random.random() < 0.8:
                norm_type = random.choice(['BatchNorm2d', 'InstanceNorm2d', 'GroupNorm'])
                if norm_type == 'GroupNorm':
                    block.append({
                        'type': norm_type,
                        'params': {
                            'num_groups': min(32, out_channels),
                            'num_channels': out_channels
                        }
                    })
                else:
                    block.append({
                        'type': norm_type,
                        'params': {'num_features': out_channels}
                    })
            
            # 3. Activation
            block.append({
                'type': random.choice(['ReLU', 'LeakyReLU', 'ELU', 'GELU']),
                'params': {}
            })
            
            # 4. Pooling (optional)
            if random.random() < 0.5:
                pool_type = random.choice(['MaxPool2d', 'AvgPool2d'])
                pool_size = random.choice([2, 3])
                block.append({
                    'type': pool_type,
                    'params': {
                        'kernel_size': pool_size,
                        'stride': pool_size
                    }
                })
                current_dim //= pool_size
            
            layers.extend(block)
            current_channels = out_channels
        
        # Add flatten layer for transition to dense layers
        layers.append({'type': 'Flatten', 'params': {}})
        return layers

    def _create_residual_backbone(self, input_dim, in_channels):
        """Create ResNet-style architecture with skip connections."""
        layers = []
        skip_connections = []
        current_dim = input_dim
        current_channels = in_channels
        
        num_residual_blocks = random.randint(2, 4)
        for block_idx in range(num_residual_blocks):
            # Store the start of the residual block
            block_start = len(layers)
            
            # Main path
            out_channels = random.randint(64, 256)
            layers.extend([
                {
                    'type': 'Conv2d',
                    'params': {
                        'in_channels': current_channels,
                        'out_channels': out_channels,
                        'kernel_size': 3,
                        'padding': 1
                    }
                },
                {'type': 'BatchNorm2d', 'params': {'num_features': out_channels}},
                {'type': 'ReLU', 'params': {}},
                {
                    'type': 'Conv2d',
                    'params': {
                        'in_channels': out_channels,
                        'out_channels': out_channels,
                        'kernel_size': 3,
                        'padding': 1
                    }
                },
                {'type': 'BatchNorm2d', 'params': {'num_features': out_channels}}
            ])
            
            # Add skip connection
            skip_connections.append({
                'from': block_start,
                'to': len(layers),
                'type': 'add'
            })
            
            current_channels = out_channels
        
        layers.append({'type': 'Flatten', 'params': {}})
        return layers

    def _create_transformer_backbone(self, input_dim):
        """Create Transformer-based architecture."""
        layers = []
        
        # Project input to transformer dimension
        d_model = random.choice([128, 256, 512])
        layers.extend([
            {
                'type': 'Linear',
                'params': {
                    'in_features': input_dim,
                    'out_features': d_model
                }
            },
            {'type': 'LayerNorm', 'params': {'normalized_shape': d_model}}
        ])
        
        # Add transformer layers
        num_layers = random.randint(2, 6)
        num_heads = random.choice([4, 8, 16])
        
        for _ in range(num_layers):
            layers.append({
                'type': 'TransformerEncoderLayer',
                'params': {
                    'd_model': d_model,
                    'nhead': num_heads,
                    'dim_feedforward': d_model * 4,
                    'dropout': 0.1
                }
            })
        
        return layers

    def _create_inception_backbone(self, input_dim, in_channels):
        """Create Inception-style architecture with parallel paths."""
        layers = []
        current_channels = in_channels
        
        num_inception_blocks = random.randint(2, 4)
        for _ in range(num_inception_blocks):
            # Create parallel paths
            paths = []
            total_channels = 0
            
            # 1x1 convolution path
            out_channels_1x1 = random.randint(32, 128)
            paths.append([{
                'type': 'Conv2d',
                'params': {
                    'in_channels': current_channels,
                    'out_channels': out_channels_1x1,
                    'kernel_size': 1
                }
            }])
            total_channels += out_channels_1x1
            
            # 3x3 convolution path
            out_channels_3x3 = random.randint(32, 128)
            paths.append([
                {
                    'type': 'Conv2d',
                    'params': {
                        'in_channels': current_channels,
                        'out_channels': out_channels_3x3,
                        'kernel_size': 3,
                        'padding': 1
                    }
                }
            ])
            total_channels += out_channels_3x3
            
            # 5x5 convolution path
            out_channels_5x5 = random.randint(16, 64)
            paths.append([
                {
                    'type': 'Conv2d',
                    'params': {
                        'in_channels': current_channels,
                        'out_channels': out_channels_5x5,
                        'kernel_size': 5,
                        'padding': 2
                    }
                }
            ])
            total_channels += out_channels_5x5
            
            # Pool path
            out_channels_pool = random.randint(16, 64)
            paths.append([
                {
                    'type': 'MaxPool2d',
                    'params': {
                        'kernel_size': 3,
                        'stride': 1,
                        'padding': 1
                    }
                },
                {
                    'type': 'Conv2d',
                    'params': {
                        'in_channels': current_channels,
                        'out_channels': out_channels_pool,
                        'kernel_size': 1
                    }
                }
            ])
            total_channels += out_channels_pool
            
            # Add parallel paths and concatenation
            layers.append({
                'type': 'InceptionBlock',
                'params': {
                    'paths': paths,
                    'out_channels': total_channels
                }
            })
            
            current_channels = total_channels
        
        layers.append({'type': 'Flatten', 'params': {}})
        return layers

    def _get_final_dim(self, layers):
        """
        Calculate the final dimension after all layers.
        """
        current_dim = 784  # MNIST input (28x28)
        current_channels = 1
        is_conv = False
        
        for layer in layers:
            layer_type = layer['type']
            params = layer.get('params', {})
            
            if layer_type == 'Flatten':
                if is_conv:
                    current_dim = current_dim * current_dim * current_channels
                is_conv = False
            
            elif layer_type.startswith('Conv'):
                is_conv = True
                current_channels = params['out_channels']
                kernel_size = params['kernel_size']
                stride = params.get('stride', 1)
                padding = params.get('padding', 0)
                current_dim = ((current_dim + 2 * padding - kernel_size) // stride + 1)
                
            elif layer_type == 'Linear':
                current_dim = params['out_features']
                
            elif layer_type in ['MaxPool2d', 'AvgPool2d']:
                kernel_size = params['kernel_size']
                stride = params.get('stride', kernel_size)
                current_dim = (current_dim // stride)
                
            elif layer_type == 'AdaptiveAvgPool2d':
                output_size = params.get('output_size', 1)
                current_dim = output_size * output_size * current_channels
                is_conv = False
                
            elif layer_type == 'InceptionBlock':
                current_channels = params['out_channels']
                
            elif layer_type == 'TransformerEncoderLayer':
                current_dim = params['d_model']
                
        return current_dim

    def _create_mlp_backbone(self, input_dim):
        """Create MLP-based architecture."""
        layers = []
        current_dim = input_dim
        
        # Flatten input
        layers.append({'type': 'Flatten', 'params': {}})
        
        # Add dense layers with varying widths
        num_layers = random.randint(2, 6)
        for i in range(num_layers):
            width = random.randint(64, 512)
            
            layers.extend([
                {
                    'type': 'Linear',
                    'params': {
                        'in_features': current_dim,
                        'out_features': width
                    }
                },
                {
                    'type': random.choice(['ReLU', 'LeakyReLU', 'ELU', 'GELU']),
                    'params': {}
                }
            ])
            
            # Add regularization
            if random.random() < 0.5:
                if random.random() < 0.7:
                    layers.append({
                        'type': 'Dropout',
                        'params': {'p': random.uniform(0.1, 0.5)}
                    })
                else:
                    layers.append({
                        'type': 'BatchNorm1d',
                        'params': {'num_features': width}
                    })
            
            current_dim = width
        
        return layers

    def _create_dense_backbone(self, input_dim, in_channels):
        """Create DenseNet-style architecture."""
        layers = []
        current_channels = in_channels
        growth_rate = random.randint(12, 32)
        
        num_dense_blocks = random.randint(2, 4)
        for block in range(num_dense_blocks):
            num_layers = random.randint(3, 6)
            
            for layer in range(num_layers):
                # Batch Norm - ReLU - Conv1x1 - Conv3x3
                layers.extend([
                    {
                        'type': 'BatchNorm2d',
                        'params': {'num_features': current_channels}
                    },
                    {
                        'type': 'ReLU',
                        'params': {}
                    },
                    {
                        'type': 'Conv2d',
                        'params': {
                            'in_channels': current_channels,
                            'out_channels': growth_rate * 4,
                            'kernel_size': 1,
                            'stride': 1,
                            'padding': 0
                        }
                    },
                    {
                        'type': 'Conv2d',
                        'params': {
                            'in_channels': growth_rate * 4,
                            'out_channels': growth_rate,
                            'kernel_size': 3,
                            'stride': 1,
                            'padding': 1
                        }
                    }
                ])
                current_channels += growth_rate
            
            # Transition layer
            if block < num_dense_blocks - 1:
                out_channels = current_channels // 2
                layers.extend([
                    {
                        'type': 'BatchNorm2d',
                        'params': {'num_features': current_channels}
                    },
                    {
                        'type': 'Conv2d',
                        'params': {
                            'in_channels': current_channels,
                            'out_channels': out_channels,
                            'kernel_size': 1,
                            'stride': 1
                        }
                    },
                    {
                        'type': 'AvgPool2d',
                        'params': {
                            'kernel_size': 2,
                            'stride': 2
                        }
                    }
                ])
                current_channels = out_channels
        
        layers.append({'type': 'Flatten', 'params': {}})
        return layers

    def _create_hybrid_backbone(self, input_dim, in_channels):
        """Create hybrid architecture combining different patterns."""
        layers = []
        current_dim = input_dim
        current_channels = in_channels
        
        # Start with convolution feature extraction
        layers.extend(self._create_cnn_backbone(current_dim // 2, current_channels))
        
        # Add transformer layers for global context
        d_model = random.choice([128, 256])
        num_heads = random.choice([4, 8])
        layers.extend([
            {
                'type': 'TransformerEncoderLayer',
                'params': {
                    'd_model': d_model,
                    'nhead': num_heads,
                    'dim_feedforward': d_model * 4,
                    'dropout': 0.1
                }
            }
        ])
        
        # Add MLP head
        layers.extend(self._create_mlp_backbone(d_model))
        
        return layers

    def _create_attention_backbone(self, input_dim):
        """Create attention-based architecture."""
        layers = []
        
        # Project to attention dimension
        attention_dim = random.choice([128, 256, 512])
        layers.extend([
            {
                'type': 'Linear',
                'params': {
                    'in_features': input_dim,
                    'out_features': attention_dim
                }
            },
            {
                'type': 'LayerNorm',
                'params': {'normalized_shape': attention_dim}
            }
        ])
        
        # Add self-attention layers
        num_attention_layers = random.randint(2, 4)
        for _ in range(num_attention_layers):
            layers.append({
                'type': 'MultiheadAttention',
                'params': {
                    'embed_dim': attention_dim,
                    'num_heads': random.choice([4, 8]),
                    'dropout': 0.1
                }
            })
            
            # Add feed-forward network after attention
            layers.extend([
                {
                    'type': 'Linear',
                    'params': {
                        'in_features': attention_dim,
                        'out_features': attention_dim * 4
                    }
                },
                {'type': 'GELU', 'params': {}},
                {
                    'type': 'Linear',
                    'params': {
                        'in_features': attention_dim * 4,
                        'out_features': attention_dim
                    }
                },
                {
                    'type': 'LayerNorm',
                    'params': {'normalized_shape': attention_dim}
                }
            ])
        
        return layers

    def _create_recurrent_backbone(self, input_dim):
        """Create recurrent-based architecture."""
        layers = []
        current_dim = input_dim
        
        # Project input to sequence
        seq_length = 28  # For MNIST, treat as 28 sequences of 28 features
        feature_dim = current_dim // seq_length
        
        # Add recurrent layers
        num_layers = random.randint(1, 3)
        hidden_dim = random.randint(64, 256)
        
        rnn_type = random.choice(['LSTM', 'GRU'])
        layers.append({
            'type': rnn_type,
            'params': {
                'input_size': feature_dim,
                'hidden_size': hidden_dim,
                'num_layers': num_layers,
                'batch_first': True,
                'dropout': 0.1 if num_layers > 1 else 0,
                'bidirectional': random.choice([True, False])
            }
        })
        
        # Add attention to RNN outputs (optional)
        if random.random() < 0.5:
            layers.append({
                'type': 'MultiheadAttention',
                'params': {
                    'embed_dim': hidden_dim * (2 if layers[-1]['params']['bidirectional'] else 1),
                    'num_heads': random.choice([4, 8]),
                    'dropout': 0.1
                }
            })
        
        return layers

    def generate_layer_params(self, layer_type, input_size):
        """
        Generate realistic parameters for a given layer type.
        """
        params = {}
        try:
            if layer_type == 'Linear':
                params['in_features'] = input_size
                params['out_features'] = random.randint(32, 1024)
            elif layer_type.startswith('Conv'):
                params['in_channels'] = input_size if 'in_channels' not in vars(self) else self.in_channels
                params['out_channels'] = random.randint(16, 256)
                params['kernel_size'] = random.choice([1, 3, 5, 7])
                params['stride'] = random.choice([1, 2])
                params['padding'] = random.choice([0, 1, 2, 3])
                self.in_channels = params['out_channels']
            elif layer_type.startswith('BatchNorm'):
                params['num_features'] = input_size
            elif layer_type in ['ReLU', 'Sigmoid', 'Tanh', 'Softmax', 'Dropout', 'ELU', 'LeakyReLU', 'ReLU6', 'PReLU', 'CELU', 'GELU', 'SELU', 'GLU']:
                pass  # No parameters required
            elif layer_type in ['LSTM', 'GRU', 'RNN', 'RNNCell', 'LSTMCell', 'GRUCell']:
                params['input_size'] = input_size
                params['hidden_size'] = random.randint(32, 512)
                params['num_layers'] = random.randint(1, 3)
                params['batch_first'] = True
            elif layer_type == 'Embedding':
                params['num_embeddings'] = random.randint(1000, 10000)
                params['embedding_dim'] = random.randint(32, 256)
                input_size = params['embedding_dim']
            elif layer_type.endswith('Pool2d') or layer_type.endswith('Pool1d') or layer_type.endswith('Pool3d'):
                params['kernel_size'] = random.choice([2, 3, 5])
                params['stride'] = params['kernel_size']
            elif layer_type.startswith('AdaptiveAvgPool') or layer_type.startswith('AdaptiveMaxPool'):
                params['output_size'] = random.choice([1, 2, 4, 8])
            elif layer_type == 'Flatten':
                pass  # No parameters required
            elif layer_type == 'Upsample':
                params['scale_factor'] = random.choice([2, 3, 4])
                params['mode'] = random.choice(['nearest', 'linear', 'bilinear', 'trilinear'])
            elif layer_type == 'ConvTranspose2d':
                params['in_channels'] = input_size if 'in_channels' not in vars(self) else self.in_channels
                params['out_channels'] = random.randint(16, 256)
                params['kernel_size'] = random.choice([2, 3, 4])
                params['stride'] = random.choice([1, 2])
                params['padding'] = random.choice([0, 1])
                self.in_channels = params['out_channels']
            elif layer_type == 'LayerNorm':
                params['normalized_shape'] = input_size
            elif layer_type == 'Transformer':
                params['d_model'] = input_size
                params['nhead'] = random.choice([2, 4, 8])
                params['num_encoder_layers'] = random.randint(1, 6)
                params['num_decoder_layers'] = random.randint(1, 6)
                params['dim_feedforward'] = random.randint(128, 1024)
                params['dropout'] = random.uniform(0.1, 0.3)
            elif layer_type == 'TransformerEncoder':
                d_model_options = [128, 256, 512, 768, 1024]
                d_model = random.choice(d_model_options)
                nhead_options = [n for n in [2, 4, 8, 16] if d_model % n == 0]
                if not nhead_options:
                    print(f"Cannot find suitable nhead for d_model {d_model}")
                    return None
                nhead = random.choice(nhead_options)
                params['d_model'] = d_model
                params['nhead'] = nhead
                params['dim_feedforward'] = random.randint(128, 2048)
                params['dropout'] = random.uniform(0.1, 0.3)
                params['activation'] = 'relu'
                params['num_layers'] = random.randint(1, 6)
            elif layer_type == 'MultiheadAttention':
                params['embed_dim'] = input_size
                params['num_heads'] = random.choice([2, 4, 8])
            elif layer_type == 'Sequential':
                # We can recursively generate a small sequential module
                sub_layers = []
                num_sub_layers = random.randint(2, 5)
                sub_input_size = input_size
                for _ in range(num_sub_layers):
                    sub_layer_info = random.choice(self.available_layers)
                    sub_layer_type = sub_layer_info['type']
                    sub_params = self.generate_layer_params(sub_layer_type, sub_input_size)
                    if sub_params is None:
                        continue
                    sub_layers.append({'type': sub_layer_type, 'params': sub_params})
                    sub_input_size = self.update_input_size(sub_layer_type, sub_params, sub_input_size)
                params['layers'] = sub_layers
            else:
                return None  # Unsupported layer type for random parameter generation
            return params
        except Exception as e:
            print(f"Error generating parameters for layer {layer_type}: {e}")
            return None


    def update_input_size(self, layer_type, params, current_size):
        """
        Update the input size based on the layer's output size.
        """
        if layer_type == 'Linear':
            return params['out_features']
        elif layer_type == 'Conv2d':
            # Simplified output size calculation
            kernel_size = params['kernel_size']
            stride = params['stride']
            padding = params['padding']
            output_size = int((current_size + 2 * padding - kernel_size) / stride + 1)
            return max(output_size, 1)  # Prevent non-positive sizes
        elif layer_type == 'LSTM':
            return params['hidden_size']
        else:
            return current_size  # For layers that don't change the size
        
    def generate_model(self, config):
        """
        Dynamically create a PyTorch model from a configuration dictionary.
        """
        class InceptionBlock(nn.Module):
            def __init__(self, paths, out_channels):
                super().__init__()
                self.paths = nn.ModuleList()
                for path in paths:
                    layers = []
                    for layer_info in path:
                        layer_type = layer_info['type']
                        params = layer_info.get('params', {})
                        LayerClass = getattr(nn, layer_type)
                        layers.append(LayerClass(**params))
                    self.paths.append(nn.Sequential(*layers))

            def forward(self, x):
                path_outputs = [path(x) for path in self.paths]
                return torch.cat(path_outputs, dim=1)

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                # Shortcut connection
                self.shortcut = nn.Sequential()
                if in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.BatchNorm2d(out_channels)
                    )
                
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out

        class CustomTransformerBlock(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
            def forward(self, x):
                x = x.reshape(x.size(0), -1, x.size(-1))  # Reshape for attention
                attn_output, _ = self.self_attn(x, x, x)
                x = self.norm1(x + attn_output)
                ff_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
                x = self.norm2(x + ff_output)
                return x

        layers = []
        available_layer_dict = {
            'InceptionBlock': InceptionBlock,
            'ResidualBlock': ResidualBlock,
            'CustomTransformerBlock': CustomTransformerBlock,
            **{layer['type']: layer['class'] for layer in self.available_layers}
        }

        current_channels = 1  # MNIST starts with 1 channel
        current_dim = 28  # MNIST image size

        for layer_info in config.get("layers", []):
            layer_type = layer_info["type"]
            params = layer_info.get("params", {})
            
            # Handle special cases for dimension tracking
            if layer_type == 'Conv2d':
                params['in_channels'] = current_channels
                current_channels = params['out_channels']
                kernel = params['kernel_size']
                stride = params.get('stride', 1)
                padding = params.get('padding', 0)
                current_dim = ((current_dim + 2*padding - kernel) // stride + 1)
                
            elif layer_type == 'Linear':
                if len(layers) > 0 and isinstance(layers[-1], nn.Conv2d):
                    params['in_features'] = current_channels * current_dim * current_dim
                
            elif layer_type in ['MaxPool2d', 'AvgPool2d']:
                kernel = params['kernel_size']
                stride = params.get('stride', kernel)
                current_dim = current_dim // stride
                
            elif layer_type == 'Flatten':
                if current_channels > 1:
                    params['start_dim'] = 1
                    
            LayerClass = available_layer_dict.get(layer_type)
            if LayerClass:
                try:
                    if layer_type == 'InceptionBlock':
                        layer = InceptionBlock(**params)
                        current_channels = params['out_channels']
                    elif layer_type == 'ResidualBlock':
                        layer = ResidualBlock(current_channels, params['out_channels'])
                        current_channels = params['out_channels']
                    elif layer_type == 'CustomTransformerBlock':
                        layer = CustomTransformerBlock(**params)
                    else:
                        layer = self.instantiate_layer(LayerClass, params)
                    
                    if layer is not None:
                        layers.append(layer)
                    else:
                        print(f"Failed to instantiate layer {layer_type}")
                        return None
                except Exception as e:
                    print(f"Error instantiating {layer_type}: {str(e)}")
                    return None
            else:
                print(f"Unsupported layer type: {layer_type}")
                return None

        try:
            model = nn.Sequential(*layers)
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 1, 28, 28)  # MNIST dimensions
            _ = model(dummy_input)
            return model
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return None

    def save_generated_model(self, config, model_id, save_path):
        """
        Save the generated model configuration to a JSON file.
        """
        with open(os.path.join(save_path, f"{model_id}.json"), 'w') as f:
            json.dump(config, f, indent=4)

    def instantiate_layer(self, LayerClass, params):
        """
        Instantiate a PyTorch layer with given parameters.
        """
        try:
            if LayerClass.__name__ in ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ReLU6', 'PReLU', 'ELU', 'SELU', 'GELU', 'Softmax', 'GLU', 'Dropout', 'Flatten']:
                return LayerClass()
            
            elif LayerClass.__name__ in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm', 'Embedding']:
                return LayerClass(**params)
            
            elif LayerClass.__name__ in ['MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d']:
                return LayerClass(kernel_size=params['kernel_size'], stride=params.get('stride', None))
            
            elif LayerClass.__name__ in ['AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d']:
                return LayerClass(output_size=params.get('output_size', 1))
            
            elif LayerClass.__name__ in ['LSTM', 'GRU', 'RNN']:
                return LayerClass(
                    input_size=params['input_size'],
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    batch_first=params.get('batch_first', True)
                )
            
            elif LayerClass.__name__ == 'Transformer':
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    dim_feedforward=params['dim_feedforward'],
                    dropout=params.get('dropout', 0.1)
                )
                return nn.TransformerEncoder(encoder_layer, num_layers=params['num_layers'])
            
            elif LayerClass.__name__ == 'MultiheadAttention':
                return LayerClass(
                    embed_dim=params['embed_dim'],
                    num_heads=params['num_heads'],
                    dropout=params.get('dropout', 0.0)
                )
            
            elif LayerClass.__name__ == 'Upsample':
                return LayerClass(
                    scale_factor=params['scale_factor'],
                    mode=params.get('mode', 'nearest')
                )
            
            else:
                print(f"Warning: Layer type {LayerClass.__name__} not explicitly handled")
                return LayerClass(**params)
            
        except Exception as e:
            print(f"Error instantiating {LayerClass.__name__}: {str(e)}")
            return None

if __name__ == "__main__":
    generator = ModelGenerator()
    num_models_to_generate = 1000  # Adjust as needed
    save_dir = 'models/generated/'
    os.makedirs(save_dir, exist_ok=True)

    for _ in range(num_models_to_generate):
        config, model_id = generator.generate_unique_model_config()
        generator.save_generated_model(config, model_id, save_dir)
        print(f"Generated model {model_id}")
