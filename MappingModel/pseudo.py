import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example: Assume you have collected data in arrays.
# X: [isolated_energy, concurrent_tasks] for each measurement.
# y: Measured energy consumption under concurrency.
X = np.array([
    [10.0, 1],
    [10.0, 2],
    [10.0, 3],
    [12.0, 1],
    [12.0, 2],
    [12.0, 3],
    # ... more data
])
y = np.array([10.0, 18.0, 25.0, 12.0, 22.0, 30.0])  # Example values

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model.
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Training R^2:", model.score(X_train, y_train))
print("Test R^2:", model.score(X_test, y_test))

# Now, given an isolated energy measurement and a concurrency level, predict the effective energy consumption.
isolated_energy = 11.0
num_concurrent = 4
predicted_energy = model.predict(np.array([[isolated_energy, num_concurrent]]))
print("Predicted energy consumption:", predicted_energy)
