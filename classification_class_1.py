import pandas as pd
import matplotlib.pyplot as plt

# Load the pickle file
data = pd.read_pickle('metrics_data.pkl')

# Define Class 1 speakers
class_1 = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

for i, metrics_data in enumerate(data):
    classification_report = metrics_data['classification_report']
    
    metrics_df = pd.DataFrame(classification_report).transpose()
    
    metrics_df = metrics_df[~metrics_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    
    metrics_df['class'] = metrics_df.index.map(lambda x: 'Class 1' if x in class_1 else 'Class 0')
    
    class_1_df = metrics_df[metrics_df['class'] == 'Class 1']
    if not class_1_df.empty:
        class_1_metrics = class_1_df[['precision', 'recall', 'f1-score']].mean() * 100
    else:
        print(f"No data available for Class 1 in epoch {i+1}.")
        class_1_metrics = pd.Series({'precision': 0, 'recall': 0, 'f1-score': 0})
    
    ax = class_1_metrics.plot(kind='bar', figsize=(8, 5), width=0.7, color=['skyblue', 'orange', 'green'])
    plt.title(f'Precision, Recall, and F1-Score for Class 1 (Epoch {i+1})')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Metric')
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() + 0.02, p.get_height() / 2), 
                    ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
_