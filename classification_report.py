import pandas as pd
import matplotlib.pyplot as plt

# Load the pickle file
data = pd.read_pickle('metrics_data.pkl')

# Iterate over each classification report in the data
for i, metrics_data in enumerate(data):
    classification_report = metrics_data['classification_report']
    
    # Extract metrics into a DataFrame
    metrics_df = pd.DataFrame(classification_report).transpose()
    
    # Filter out non-class-specific rows like 'accuracy', 'macro avg', and 'weighted avg'
    metrics_df = metrics_df[~metrics_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    

    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(15, 6))
    plt.title(f'Precision, Recall, and F1-Score for Data Set {i+1}')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()  
