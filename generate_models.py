import argparse
from pipeline.model_generator import ModelGenerator
import os

def generate_and_save_models(num_models, save_dir):
    generator = ModelGenerator()
    os.makedirs(save_dir, exist_ok=True)
    for _ in range(num_models):
        config, model_id = generator.generate_unique_model_config()
        generator.save_generated_model(config, model_id, save_dir)
        print(f"Generated model {model_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random model architectures')
    parser.add_argument('--num_models', type=int, default=100, help='Number of models to generate')
    parser.add_argument('--save_dir', type=str, default='models/generated/', help='Directory to save generated models')
    args = parser.parse_args()
    generate_and_save_models(args.num_models, args.save_dir)
