import numpy as np
from sklearn.linear_model import LinearRegression

# Training the dog with treat examples
X = np.array([[1], [2], [3], [4], [5]])  # Number of treats
y = np.array([1, 2, 3, 4, 5])            # Number of sits

# Teach the dog the rule
model = LinearRegression()
model.fit(X, y)

# Ask the dog: "How many times will you sit with 6 treats?"
prediction = model.predict([[6]])

# Show the answer
print(prediction[0])  # Output: 6
