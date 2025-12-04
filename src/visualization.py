import matplotlib.pyplot as plt


models = ["TF-IDF", "Word2Vec", "Doc2Vec", "BERT", "RoBERTa", "SBERT", "FairHire-AI"]

accuracy = [72.4, 76.8, 79.1, 86.3, 87.9, 89.4, 92.1]
precision = [70.1, 74.3, 77.8, 84.9, 86.4, 88.2, 90.7]
recall = [68.8, 72.9, 75.2, 83.5, 85.1, 87.1, 89.4]
f1 = [69.4, 73.5, 76.0, 84.2, 85.7, 87.6, 90.0]

SAVE_DIR = "visualizations/"

import os
os.makedirs(SAVE_DIR, exist_ok=True)



def create_bar_graph(metric_values, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(models, metric_values)
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + filename)
    plt.close()



def create_line_graph(metric_values, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(models, metric_values, marker='o')
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SAVE_DIR + filename)
    plt.close()





create_bar_graph(accuracy, "Model Comparison - Accuracy", "Accuracy (%)", "accuracy_bar.png")
create_bar_graph(precision, "Model Comparison - Precision", "Precision (%)", "precision_bar.png")
create_bar_graph(recall, "Model Comparison - Recall", "Recall (%)", "recall_bar.png")
create_bar_graph(f1, "Model Comparison - F1 Score", "F1 Score (%)", "f1_bar.png")


create_line_graph(accuracy, "Model Comparison - Accuracy (Line Graph)", "Accuracy (%)", "accuracy_line.png")
create_line_graph(precision, "Model Comparison - Precision (Line Graph)", "Precision (%)", "precision_line.png")
create_line_graph(recall, "Model Comparison - Recall (Line Graph)", "Recall (%)", "recall_line.png")
create_line_graph(f1, "Model Comparison - F1 Score (Line Graph)", "F1 Score (%)", "f1_line.png")

print("Visualization graphs generated successfully in /visualizations folder!")