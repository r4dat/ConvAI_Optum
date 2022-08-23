from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

features = data["data"]
labels = data["target"]

model = DecisionTreeClassifier()

model.fit(features, labels)

predictions = model.predict(features)

print(predictions)