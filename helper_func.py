import torch
from sklearn.metrics import (accuracy_score, recall_score, precision_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
import numpy as np

# Training, test and evaluation functions
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    for X, y in data_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            loss = loss_function(output, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def get_predictions(data_loader, model):
    y_pred = []
    y_test = []
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            y_pred.extend(output.numpy())  # Logits
            y_test.extend(y.numpy())       # True labels
    return y_pred, y_test


def get_metrics(outputs, targets):
    # Convert outputs to probabilities
    probabilities = torch.sigmoid(torch.tensor(outputs)).numpy()

    # Compute optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]

    # Apply threshold to get binary predictions
    predictions = (probabilities >= optimal_threshold).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(targets, predictions)
    recall = recall_score(targets, predictions, zero_division=0)
    precision = precision_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    auc_score = roc_auc_score(targets, probabilities)
    confusion = confusion_matrix(targets, predictions)

    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'auc_score': auc_score,
        'optimal_threshold': optimal_threshold,
        'confusion': confusion
    }

    return metrics

def matrix_eval(outputs, targets):
    # Get evaluation metrics
    metrics = get_metrics(outputs, targets)
    print(f"\nOptimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"AUC-ROC: {metrics['auc_score']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion']}")