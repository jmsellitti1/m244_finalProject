import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_parquet("data/2025_hits.parquet")
X = data[['launch_speed', 'launch_angle', 'hit_distance',
          'hit_location_x', 'hit_location_y',
          'pitch_speed', 'release_spin_rate']]
y = data['xBA']

epsilon = 1e-6
y_clipped = np.clip(y, epsilon, 1 - epsilon)
y_logit = np.log(y_clipped / (1 - y_clipped))

X_train, X_test, y_train, _ = train_test_split(X, y_logit, test_size=0.2, random_state=42)

# Get original y values for test set using the same indices
_, _, _, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg', LinearRegression())
])
model.fit(X_train, y_train)

y_pred_logit = model.predict(X_test)
y_pred = 1 / (1 + np.exp(-y_pred_logit)) # Sigmoid activation function

# Model Evaluation
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

output = ""

output += (f"Mean Squared Error: {mse}")
output += (f"\nR^2 Score: {r2}")

output += ("\n\n5 Most Important Model Coefficients:\n")
feature_names = model.named_steps['poly'].get_feature_names_out(X.columns)
for i in np.argsort(np.abs(model.named_steps['reg'].coef_))[-5:][::-1]:
    output += (f"{feature_names[i]}: {model.named_steps['reg'].coef_[i]:.4f}\n")

print(output)
with open("figures/regression_results.txt", "w") as f:
    f.write(output)

plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual xBA')
plt.ylabel('Predicted xBA')
plt.title('Actual vs Predicted xBA')
plt.savefig("figures/xba_scatter.png")