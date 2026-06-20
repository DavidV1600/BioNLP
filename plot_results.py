import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ef_auc", type=float, required=True)
    parser.add_argument("--ef_f1", type=float, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define models and their metrics
    models = ['Transformer (Text)', 'XGBoost (Stats)', 'Late Fusion (Avg)', 'Early Fusion (Unified)']
    
    # We use our known metrics + the new ones
    roc_aucs = [0.8210, 0.8466, 0.8486, args.ef_auc]
    f1_scores = [0.2526, 0.2972, 0.3112, args.ef_f1]

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    rects1 = ax.bar(x - width/2, roc_aucs, width, label='ROC-AUC', color='#4C72B0')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#55A868')

    # Add text, title, and custom x-axis tick labels
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison: Predicting Dosing Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')

    # Attach a text label above each bar, displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    # Save the plot
    save_path = "model_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
