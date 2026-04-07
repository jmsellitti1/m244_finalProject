import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_parquet("data/2025_hits.parquet")
X = data[['launch_speed', 'launch_angle', 'hit_distance', 'hit_location_x', 'hit_location_y', 'pitch_speed', 'release_spin_rate']]
y = data['xBA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = [max(0, min(pred, 1)) for pred in y_pred]
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual xBA')
plt.ylabel('Predicted xBA')
plt.title('Actual vs Predicted xBA')
plt.savefig("data/xba_scatter.png")