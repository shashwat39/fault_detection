import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression            # max_iter=1000
from sklearn.neighbors import KNeighborsClassifier             # n_neighbors=5
from sklearn.tree import DecisionTreeClassifier                # random_state=42
from sklearn.ensemble import RandomForestClassifier            # n_estimators=100
from sklearn.svm import SVC                                    # kernel='rbf', probability=True
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prepare directories
os.makedirs("results_ML/visualizations", exist_ok=True)

# 2. Load and shuffle data
df = pd.read_csv("Data/TrainandTestDataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Extract features + targets
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
target_cols = ["LG", "LL", "LLG", "LLL", "None"]
X = df[features].values
y = np.argmax(df[target_cols].values, axis=1)  # one-hot → single label

# 4. Add 5% Gaussian noise
noise_factor = 0.05
X_noisy = X.copy().astype(float)
for i in range(X_noisy.shape[1]):
    std_i = X_noisy[:, i].std()
    noise = np.random.normal(0, std_i * noise_factor, size=X_noisy.shape[0])
    X_noisy[:, i] += noise

# 5. Train/test split + scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF kernel)": SVC(kernel='rbf', probability=True, random_state=42),
}

# 7. Train, evaluate, collect accuracies
accuracies = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies[name] = acc
    print(f"{name:20s} test accuracy: {acc:.4f}")

# 8. Save accuracies to CSV
acc_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Test Accuracy'])
acc_df.index.name = 'Model'
acc_df.to_csv("results_ML/accuracies.csv")

# 9. Plot comparative accuracies
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(10,6))
sns.barplot(
    x=acc_df['Test Accuracy'], 
    y=acc_df.index, 
    palette="viridis",
    orient='h'
)
plt.xlim(0,1)
plt.xlabel("Test Accuracy")
plt.title("Comparison of Classical ML Model Accuracies")
plt.tight_layout()
for i, (_, row) in enumerate(acc_df.iterrows()):
    plt.text(row['Test Accuracy'] + 0.005, i, f"{row['Test Accuracy']:.2%}", va='center')
plt.savefig("results_ML/visualizations/accuracy_comparison.png")
plt.close()
