import pandas as pd
from sklearn.model_selection import train_test_split


kidney_data = pd.read_csv("data/kidney_disease.csv")

X = kidney_data.drop(columns=["classification"])

y = kidney_data["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Training set shapes:", X_train.shape, y_train.shape)
print("Testing set shapes:", X_test.shape, y_test.shape)
"""
Why shouldn't we train and aswell test on the same data?
Because you’d be grading the model on questions it already saw, so the score looks better than it really is.
The model can basically “memorize” patterns and you won’t know if it works on new data.

What is the purpose of the testing set?
The test set is kept aside and only used at the end to estimate how well the model generalizes to unseen data.
It gives a more honest performance check than evaluating on the training set.
"""