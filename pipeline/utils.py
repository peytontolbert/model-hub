import torch.nn as nn
import inspect
import os


def get_available_layers():
    """
    Retrieve a list of available layers from torch.nn and custom layers.
    """
    layer_classes = []
    excluded_layers = set([
        'Module', 'Container', 'Parameter', 'ParameterList', 'ParameterDict',
        'LazyModuleMixin', 'ConvTransposeMixin', 'BatchNorm',
        'RNNBase', 'TransformerMixin',  # Base or mixin classes
        # '_ConvNd', '_BatchNorm', '_RNNImpl', '_LazyConvXdMixin',  # Private classes
    ])
    # Get all classes in torch.nn
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            # Exclude known base classes and mixins
            if name in excluded_layers:
                continue
            layer_classes.append({'type': name, 'class': obj})

    return layer_classes

def ensure_dir(directory):
    """
    Ensure that a directory exists.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)