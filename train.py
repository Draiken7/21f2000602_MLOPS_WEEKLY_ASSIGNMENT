# train.py - Week 1 reference notebook code, as a script
import json, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("data/iris.csv")                          # current version of the data (versioned by DVC)
train, test = train_test_split(data, test_size=0.4, stratify=data["species"], random_state=42)
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train, y_train = train[features], train.species
X_test, y_test = test[features], test.species

mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)                                 # train the model
prediction = mod_dt.predict(X_test)
acc = metrics.accuracy_score(prediction, y_test)             # evaluate on the 40% test split
print(f"Rows: {len(data)} | The accuracy of the Decision Tree is {acc:.3f}")

os.makedirs("artifacts", exist_ok=True)
joblib.dump(mod_dt, "artifacts/model.joblib")                # model file (versioned by DVC)
with open("metrics.json", "w") as f:                         # small text file (versioned by Git)
    json.dump({"rows": len(data), "accuracy": round(acc, 4)}, f, indent=2)
