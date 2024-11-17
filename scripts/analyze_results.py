import os
import json
import argparse
import matplotlib.pyplot as plt

def analyze_results(results_dir):
    """
    Analyze training results and produce insights.
    """
    accuracies = []
    model_ids = []
    for result_file in os.listdir(results_dir):
        if result_file.endswith('.json'):
            with open(os.path.join(results_dir, result_file), 'r') as f:
                result = json.load(f)
                accuracies.append(result['val_accuracy'])
                model_ids.append(result['model_id'])

    # Generate a plot of model accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel('Model Index')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Performance')
    plt.savefig(os.path.join(results_dir, 'model_performance.png'))
    plt.close()

    # Print top-performing models
    top_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)[:10]
    print("Top 10 Models:")
    for idx in top_indices:
        print(f"Model ID: {model_ids[idx]}, Validation Accuracy: {accuracies[idx]:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--results_dir', type=str, default='experiments/results/', help='Directory with training results')
    args = parser.parse_args()
    analyze_results(args.results_dir)
