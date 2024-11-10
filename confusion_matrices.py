import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pickle file

data = pd.read_pickle('metrics_data.pkl')
print(data[0].keys())

for metrics_data in data:
    #metrics_data = data[0]  # Get the first dictionary in the list
    confusion_matrix_df = metrics_data['confusion_matrix']
    top_5_best_recall_df = metrics_data['top_5_best_recall']
    top_5_worst_recall_df = metrics_data['top_5_worst_recall']


    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.show()