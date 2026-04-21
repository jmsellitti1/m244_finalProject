import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_parquet("data/2025_hits.parquet")
X = data[['launch_speed', 'launch_angle', 'hit_distance',
          'hit_location_x', 'hit_location_y',
          'pitch_speed', 'release_spin_rate']]
y = data['hit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
model.fit(X_train, y_train)

# Get probabilities instead of predictions
y_probs = model.predict_proba(X_test)[:, 1]

# # Model Evaluation - Adjust threshold for classification (default is 0.5) maximizing F1 score
thresholds = [i / 100 for i in range(10, 91)]  # 0.10 → 0.90
best_threshold = None
best_f1 = -1
best_metrics = None
for t in thresholds:
    y_pred = (y_probs > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
        best_metrics = confusion_matrix(y_test, y_pred)

print(f"Best Threshold: {best_threshold}")

print("Confusion Matrix:")
print("[[TN, FP]")
print(" [FN, TP]]")
print(best_metrics)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_probs)
print(f"ROC-AUC Score: {roc_auc}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("figures/roc_curve.png")