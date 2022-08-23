from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  

data = load_iris()


features = data["data"]
labels = data["target"]

# Split 80/20 
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=7)

model = DecisionTreeClassifier()

model.fit(train_x, train_y)

predictions = model.predict(test_x)

print(predictions)
# Accuracy 
score = accuracy_score(test_y,predictions)  
print(f'Accuracy score is {score:3.3f}')
print(confusion_matrix(test_y,predictions))
plot_confusion_matrix(model,test_x,test_y)
plt.show

