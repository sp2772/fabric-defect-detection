import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import tensorflow as tf

def calculate_metrics(y_true, y_pred, threshold):
    """Calculate TP, TN, FP, FN, FPR, FNR for given threshold"""
    predictions = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1 = f1_score(y_true, predictions)
    
    return {
        'threshold': threshold,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'FPR': fpr, 'FNR': fnr, 'F1': f1
    }

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report
import tensorflow as tf
from current import create_augmented_data_generator
def adaptive_threshold_determination_multiclass1(model, test_generator):
    """
    Evaluate a multi-class classification model.
    
    Args:
        model: Trained model
        test_generator: Test data generator
    """
    # Reset generator and get predictions
    #test_generator.reset()
    #steps = len(test_generator)
    
    # Get all predictions and true labels
    y_true = []
    y_pred_probs = []
    
    for i in range(steps):
        x_batch, y_batch = next(test_generator)
        batch_pred = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot labels to class indices
        y_pred_probs.extend(batch_pred)  # Keep full probability distribution
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Convert predicted probabilities to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Get class names
    class_names = list(test_generator.class_indices.keys())
    num_classes = len(class_names)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute per-class F1 scores
    f1_scores = f1_score(y_true, y_pred, average=None)  # Per-class F1-score
    overall_f1 = f1_score(y_true, y_pred, average="macro")  # Macro-average F1-score
    
    # ROC Curve & AUC for multiclass (One-vs-Rest)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(num_classes):
        # Convert labels to binary (One-vs-Rest)
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = y_pred_probs[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_pred_binary)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro-Averaged AUC
    macro_auc = np.mean(list(roc_auc.values()))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve (Macro AUC = {macro_auc:.2f})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Print per-class F1 scores
    print("\nPer-Class F1 Scores:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {f1_scores[i]:.4f}")

    print(f"\nOverall Macro F1 Score: {overall_f1:.4f}")
    print(f"Macro-Averaged ROC AUC: {macro_auc:.4f}")

# Usage:
def evaluate_multiclass(model, test_generator):
    """
    Evaluate model on a multi-class dataset.
    """
    adaptive_threshold_determination_multiclass1(model, test_generator)
IMG_SIZE = (256, 256)  # Image size
BATCH_SIZE = 40  # Number of images in each batch
test_dir = "/mnt/c/256x256Train/test/MultiClass"
test_generator = create_augmented_data_generator(
        test_dir, 
        BATCH_SIZE, 
        IMG_SIZE, 
        color_mode='grayscale'
    )
evaluate_multiclass(model='best_model_with_sobel.keras',test_generator=test_generator)