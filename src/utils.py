import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc_roc(true_labels, predicted_labels) -> float :
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    
    # Compute AUC
    auc = roc_auc_score(true_labels, predicted_labels)
    print("AUC:", auc)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random chance line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return auc