import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/kidney_disease.csv")

df = df.replace("?", pd.NA)

y = df["classification"]

if y.dtype == object:
    y = y.astype(str).str.strip().map({"ckd": 0, "notckd": 1})

X = df.drop(columns=["classification", "id"], errors="ignore")

X = pd.get_dummies(X, drop_first=True)

X = X.apply(pd.to_numeric, errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (CKD=0):", precision_score(y_test, y_pred, pos_label=0))
print("Recall (CKD=0):", recall_score(y_test, y_pred, pos_label=0))
print("F1 (CKD=0):", f1_score(y_test, y_pred, pos_label=0))
""" 
True positive: predicted chronic kidney disease (0) and the patient truly has chronic kidney disease.
True negative: predicted not chronic kidney disease (1) and the patient truly does not have chronic kidney disease.
False positive: predicted chronic kidney disease (0) but the patient is actually not chronic kidney disease (1).
False negative: predicted not chronic kidney disease (1) but the patient actually has chronic kidney disease (0) — this is the dangerous miss.
Accuracy alone can be misleading if one class is more common, since you can get high accuracy by guessing the majority class.
If missing a chronic kidney disease case is very serious, recall for chronic kidney disease is the most important because it focuses on catching as many real chronic kidney disease cases as possible.
"""