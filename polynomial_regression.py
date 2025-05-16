import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Wine Quality dataset
# Adjust the path if needed
data = pd.read_csv("california_housing_clean.csv")

# Prepare feature matrix and target vector
feature_columns = data.columns[:-1]  # all columns except 'the last one'
X = data[feature_columns].values
y = data['MedHouseVal'].values.reshape(-1, 1)

X_poly_rows = []

for row in X:
    # Initialize an empty row for X_poly
    new_row = []

    # Add the bias term
    new_row.append(1)

    # Add the linear terms
    new_row.extend(row)

    # Add the squared terms using list comprehension
    new_row.extend([x**2 for x in row])

    # Add the interaction terms
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            new_row.append(row[i] * row[j])
    
    X_poly_rows.append(new_row)

X_poly = np.array(X_poly_rows)

# Compute the parameter vector theta using the normal equation
x_hat = np.linalg.pinv(X_poly) @ y

# Predict
y_hat = X_poly @ x_hat

# Compute residuals (errors)
e = y - y_hat  # This is the residual vector

# Evaluate performance
SSE = np.sum(e ** 2)
SST = np.sum((y - np.mean(y)) ** 2)
R_squared = 1 - (SSE / SST)
MAE = np.mean(np.abs(e))
print(f"R^2: {R_squared:.4f}")
print(f"Mean Absolute Error: {MAE:.4f}")

# 2D Plot: Actual vs Predicted Quality
plt.figure()
plt.scatter(y, y_hat, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
