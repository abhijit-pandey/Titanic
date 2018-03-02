import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn import svm
train = pd.read_csv('train_clean.csv')
train.set_index('PassengerId',inplace=True)
y = np.array(train.Survived)
X = np.array(train.drop('Survived',axis=1))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
print(clf.score(X_test,y_test))
test = pd.read_csv('test_clean.csv')
res = pd.DataFrame()
res['PassengerId'] = test.PassengerId
test.set_index('PassengerId',inplace=True)
test = test.fillna(0)
X = np.array(test)
res['Survived'] = pd.DataFrame(clf.predict(X),columns=['Survived'])
res.set_index('PassengerId',inplace=True)
res.to_csv('res.csv')