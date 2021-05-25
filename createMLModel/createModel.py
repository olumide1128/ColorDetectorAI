import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import joblib


#import the csv file
df = pd.read_csv('primaryColors.csv')


#divide into features and label

X = df[['r','g','b']]
y = df['colorname']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#fit with logistic regression
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn.fit(X_train, y_train)


#predict the test data
y_pred = knn.predict(X_test)
print(y_pred)


#Performance
#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

#Accuracy Score = (TP + TN) / (TP + TN + FP + FN)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

#predict single data
d = knn.predict([[47, 68, 173]])
print(d)

#model persistence
joblib.dump(knn, "colorDetectionModel.joblib")