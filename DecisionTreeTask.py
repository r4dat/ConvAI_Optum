from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt     

data = load_iris()


features = data["data"]
labels = data["target"]
names = data["target_names"]
print(data["target_names"])

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
    
# Test depth options
tree_depths = [1,5,10]
parameters = {'max_depth' : tree_depths,'random_state' : [3]}
cv_model = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=5,scoring='accuracy')
cv_model.fit(X=train_x, y=train_y)
final_model = cv_model.best_estimator_
print (cv_model.best_score_, cv_model.best_params_) 
final_predict = final_model.predict(test_x)
print(confusion_matrix(test_y,final_predict))
ConfusionMatrixDisplay.from_predictions(test_y,final_predict)
#plt.show()

## Label 1 (Versicolor) has the most false-positives. 10 TP, 2 FP on the hold-out set. 
## I didn't expect the same confusion matrix after CV but we've made it all deterministic so it is what it is.