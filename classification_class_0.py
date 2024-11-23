import pandas as pd
import matplotlib.pyplot as plt

# Load the pickle file
data = pd.read_pickle('metrics_data.pkl')

# Define Class 1 speakers
class_1 = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']

# Iterate over each classification report in the data
for i, metrics_data in enumerate(data):
    classification_report = metrics_data['classification_report']
    
    # Extract metrics into a DataFrame
    metrics_df = pd.DataFrame(classification_report).transpose()
    
    # Filter out non-class-specific rows like 'accuracy', 'macro avg', and 'weighted avg'
    metrics_df = metrics_df[~metrics_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    
    # Reassign classes: Group specific speakers into Class 1, others into Class 0
    metrics_df['class'] = metrics_df.index.map(lambda x: 'Class 1' if x in class_1 else 'Class 0')
    
    # Compute metrics for Class 0 only
    class_0_metrics = metrics_df[metrics_df['class'] == 'Class 0'][['precision', 'recall', 'f1-score']].mean()
    
    # Convert to percentages for better readability
    class_0_metrics *= 100  # Convert to percentages

    # Plot metrics for Class 0
    ax = class_0_metrics.plot(kind='bar', figsize=(8, 5), width=0.7, color=['skyblue', 'orange', 'green'])
    plt.title(f'Precision, Recall, and F1-Score for Class 0 (Epoch {i+1})')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Metric')
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    # Add percentage values beside bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() + 0.02, p.get_height() / 2),  # Position to the side
                    ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
