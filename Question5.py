import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Question 5

df = pd.read_csv("data/kidney_disease.csv")
df = df.replace("?", pd.NA)
df.columns = df.columns.str.strip()


y_raw = df["classification"].astype(str).str.strip().str.lower()

y = y_raw.map({"ckd": 0, "notckd": 1})

y = y.combine_first(pd.to_numeric(y_raw, errors="coerce"))

keep = y.notna()
df = df.loc[keep].copy()
y = y.loc[keep].astype(int)

print("Rows after cleaning:", len(df))
print("Label counts:\n", y.value_counts(), "\n")

X = df.drop(columns=["classification", "id"], errors="ignore")
X = pd.get_dummies(X, drop_first=True)
X = X.apply(pd.to_numeric, errors="coerce")

strat = y if (y.nunique() == 2 and y.value_counts().min() >= 2) else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=strat
)

k_values = [1, 3, 5, 7, 9]
results = []

for k in k_values:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append([k, acc])

results_table = pd.DataFrame(results, columns=["k", "test_accuracy"])
print(results_table)

best_k = results_table.loc[results_table["test_accuracy"].idxmax(), "k"]
best_acc = results_table["test_accuracy"].max()
print(f"\nBest k: {best_k} (accuracy = {best_acc})")

""" 

Changing k changes how “local” the model is: small k follows nearby points closely, large k averages over more neighbors.
Very small k can overfit because outliers can flip predictions easily.
Very large k can underfit because it smooths too much and ignores real local structure.
A middle k often balances sensitivity and stability best.

"""