import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, JaccardIndex, AUROC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, log_loss, cohen_kappa_score
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import time

# Create directories to save results
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)
os.makedirs("results/models", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare the data
print("Loading and preparing data...")
Class_train = pd.read_csv("Data/TrainandTestDataset.csv")
Class_train_suffled = shuffle(Class_train, random_state=2)

# Display dataset information
print(f"Dataset shape: {Class_train_suffled.shape}")
print(f"Sample of data:\n{Class_train_suffled.head()}")

# Define features and target columns
features_col = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
target_col = ["LG", "LL", "LLG", "LLL", "None"]
X = Class_train_suffled[features_col]
Y = Class_train_suffled[target_col]

# Visualize raw data distributions
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_col):
    plt.subplot(2, 3, i+1)
    sns.histplot(X[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig("results/visualizations/raw_data_distribution.png")
plt.close()

# Visualize correlations between features
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig("results/visualizations/feature_correlation.png")
plt.close()

# Visualize target distribution
plt.figure(figsize=(10, 6))
Y.sum().plot(kind='bar', color='skyblue')
plt.title('Distribution of Fault Types')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.savefig("results/visualizations/target_distribution.png")
plt.close()

# Add some noise to the raw data
print("Adding noise to the raw data...")
noise_factor = 0.015  # Increased to 5% noise level
X_noisy = X.copy()
for col in features_col:
    noise = np.random.normal(0, X_noisy[col].std() * noise_factor, size=X_noisy[col].shape)
    X_noisy[col] = X_noisy[col] + noise

# Visualize original vs noisy data for one feature
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(X['Ia'], kde=True, color='blue')
plt.title('Original Ia Distribution')
plt.subplot(1, 2, 2)
sns.histplot(X_noisy['Ia'], kde=True, color='red')
plt.title('Noisy Ia Distribution (5% noise)')
plt.tight_layout()
plt.savefig("results/visualizations/noise_comparison.png")
plt.close()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_noisy)

# Split the data
x_train, x_val, y_train, y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=10)

# Convert data to PyTorch tensors
x_train_tensor = torch.FloatTensor(x_train).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
x_val_tensor = torch.FloatTensor(x_val).to(device)
y_val_tensor = torch.FloatTensor(y_val.values).to(device)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create a PyTorch model
class FaultClassifier(nn.Module):
    def __init__(self):
        super(FaultClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 60),
            nn.ReLU(),
            nn.BatchNorm1d(60),
            nn.Linear(60, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.model(x)

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize model, loss function, and optimizer
model = FaultClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Print model information
num_params = count_parameters(model)
print(f"Model architecture:\n{model}")
print(f"Number of trainable parameters: {num_params:,}")

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # For tracking metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    # Initialize metric trackers
    train_accuracy = Accuracy(task='multilabel', num_labels=5, average='micro').to(device)
    val_accuracy = Accuracy(task='multilabel', num_labels=5, average='micro').to(device)
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Progress bar for training
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for inputs, targets in train_loop:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            train_accuracy(outputs, targets)
            train_losses.append(loss.item())
            
            # Update progress bar
            train_loop.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        # Progress bar for validation
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad():
            for inputs, targets in val_loop:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                val_accuracy(outputs, targets)
                val_losses.append(loss.item())
                
                # Update progress bar
                val_loop.set_postfix(loss=loss.item())
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = train_accuracy.compute().item()
        val_acc = val_accuracy.compute().item()
        
        # Reset metrics for next epoch
        train_accuracy.reset()
        val_accuracy.reset()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print train loss after every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Print detailed metrics and save model at intervals
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save intermediate model
            torch.save(model.state_dict(), f"results/models/model_epoch_{epoch+1}.pt")
            
            # Plot learning curves
            plot_learning_curves(history, epoch+1)
    
    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return history

# Function to plot learning curves
def plot_learning_curves(history, epochs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs+1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs+1), history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"results/visualizations/learning_curves_epoch_{epochs}.png")
    plt.close()

# Function to evaluate model with multiple metrics
def evaluate_model(model, val_loader, class_names):
    model.eval()
    device = next(model.parameters()).device

    # Initialize lists to store true and predicted labels
    all_targets = []
    all_outputs = []
    
    # Evaluate model
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating model"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            all_targets.append(targets)
            all_outputs.append(outputs)
    
    # Concatenate all batches
    y_true = torch.cat(all_targets, dim=0)
    y_pred_raw = torch.cat(all_outputs, dim=0)
    y_pred = (y_pred_raw > 0.5).long()  # Convert to long (integer) tensor
    y_true = y_true.long()  # Convert targets to long (integer) tensor

    # Initialize metrics on the correct device
    metrics = {
        'accuracy': Accuracy(task='multilabel', num_labels=5, average='micro').to(device),
        'precision': Precision(task='multilabel', num_labels=5, average='micro').to(device),
        'recall': Recall(task='multilabel', num_labels=5, average='micro').to(device),
        'f1': F1Score(task='multilabel', num_labels=5, average='micro').to(device),
        'mcc': MatthewsCorrCoef(task='multilabel', num_labels=5).to(device),
        'jaccard': JaccardIndex(task='multilabel', num_labels=5).to(device),
        'auroc': AUROC(task='multilabel', num_labels=5).to(device)
    }

    # Calculate metrics
    results = {}
    for name, metric in metrics.items():
        if name == 'auroc':
            results[name] = metric(y_pred_raw, y_true).item()
        else:
            results[name] = metric(y_pred, y_true).item()

    # Move to CPU for numpy operations
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_raw_np = y_pred_raw.cpu().numpy()

    # Calculate additional metrics
    results['hamming_loss'] = hamming_loss(y_true_np, y_pred_np)
    results['log_loss'] = log_loss(y_true_np, y_pred_raw_np)
    
    # Convert to one-class per sample for Cohen's Kappa
    y_true_class = np.argmax(y_true_np, axis=1)
    y_pred_class = np.argmax(y_pred_np, axis=1)
    results['cohen_kappa'] = cohen_kappa_score(y_true_class, y_pred_class)

    # Print and save metrics
    print("\nModel Evaluation Metrics:")
    for name, value in results.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")

    # Save metrics to file
    with open("results/metrics/evaluation_metrics.txt", 'w') as f:
        f.write("Model Evaluation Metrics:\n")
        for name, value in results.items():
            f.write(f"{name.replace('_', ' ').title()}: {value:.4f}\n")

    # Create and save confusion matrix
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(len(class_names), len(class_names))
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='crest')
    plt.xlabel('Predicted Fault Type', fontsize=12)
    plt.ylabel('Actual Fault Type', fontsize=13)
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names)
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/visualizations/confusion_matrix.png")
    plt.close()

    return results, cm

# Plot additional visualizations
def plot_additional_visualizations(model, x_val_tensor, y_val_tensor, class_names):
    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_pred_raw = model(x_val_tensor)
        y_pred = (y_pred_raw > 0.5).float()
    
    # Move to CPU for numpy operations
    y_val_np = y_val_tensor.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_pred_raw_np = y_pred_raw.cpu().numpy()
    
    # 1. Prediction confidence histogram
    plt.figure(figsize=(10, 6))
    for i, cls in enumerate(class_names):
        plt.hist(y_pred_raw_np[:, i], bins=20, alpha=0.5, label=cls)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/visualizations/prediction_confidence.png")
    plt.close()
    
    # 2. Correct vs Incorrect predictions
    correct_mask = (y_pred_np == y_val_np).all(axis=1)
    incorrect_mask = ~correct_mask
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Correct', 'Incorrect'], [sum(correct_mask), sum(incorrect_mask)], color=['green', 'red'])
    plt.title('Correct vs Incorrect Predictions')
    plt.ylabel('Count')
    for i, v in enumerate([sum(correct_mask), sum(incorrect_mask)]):
        plt.text(i, v + 0.1, f"{v} ({v/len(correct_mask):.1%})", ha='center')
    plt.tight_layout()
    plt.savefig("results/visualizations/correct_vs_incorrect.png")
    plt.close()
    
    # 3. Per-class accuracy
    class_accuracies = []
    for i, cls in enumerate(class_names):
        class_acc = np.mean((y_pred_np[:, i] == y_val_np[:, i]).astype(float))
        class_accuracies.append(class_acc)
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_accuracies, color='skyblue')
    plt.title('Per-Class Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(class_accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("results/visualizations/per_class_accuracy.png")
    plt.close()
    
    # 4. ROC Curves
    plt.figure(figsize=(10, 8))
    from sklearn.metrics import roc_curve, auc
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_val_np[:, i], y_pred_raw_np[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/visualizations/roc_curves.png")
    plt.close()
    
    # 5. Feature importance visualization
    # Create a simple feature importance analysis by examining the first layer weights
    with torch.no_grad():
        # Get the weights of the first layer
        first_layer_weights = model.model[0].weight.cpu().numpy()
        
        # Take the absolute mean across all output neurons to get feature importance
        feature_importance = np.abs(first_layer_weights).mean(axis=0)
        
        # Normalize to sum to 100%
        feature_importance = feature_importance / feature_importance.sum() * 100
        
        plt.figure(figsize=(10, 6))
        plt.bar(features_col, feature_importance, color='teal')
        plt.title('Feature Importance (Based on First Layer Weights)')
        plt.xlabel('Features')
        plt.ylabel('Importance (%)')
        for i, v in enumerate(feature_importance):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
        plt.tight_layout()
        plt.savefig("results/visualizations/feature_importance.png")
        plt.close()

# Run the training
print("\nStarting model training...")
class_names = ["LG", "LL", "LLG", "LLL", "None"]
history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000)

# Save the final model
torch.save(model.state_dict(), "results/models/final_model.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history': history
}, "results/models/model_checkpoint.pth")

# Evaluate model
print("\nEvaluating final model...")
metrics, confusion_mat = evaluate_model(model, val_loader, class_names)

# Plot additional visualizations
print("\nGenerating additional visualizations...")
plot_additional_visualizations(model, x_val_tensor, y_val_tensor, class_names)

# Plot final learning curves
plot_learning_curves(history, len(history['train_loss']))

print("\nTraining and evaluation completed!")
print(f"All results saved to the 'results' directory.")