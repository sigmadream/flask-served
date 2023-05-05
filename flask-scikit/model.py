import pandas as pd
import numpy as np

url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
dataset = pd.read_csv(url)

X = dataset[['Sex', 'Age', 'Pclass']].values
y = dataset['Survived'].values
print(X)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [1]])
X[:, [1]] = imputer.transform(X[:, [1]])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

import joblib
joblib.dump(classifier, './model/model.pkl')
print("Model sucesfully dumped")

joblib.dump(sc, './model/sc.pkl')
joblib.dump(ct, './model/ct.pkl')