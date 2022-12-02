import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('./water_potability.csv')

df.fillna(df.mean(), inplace=True)

X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values

Y = df.iloc[:, [9]].values

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=101)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
Y_train = Y_train.astype('int')
classifier.fit(X_train, Y_train)
pickle.dump(classifier, open('model.pkl','wb'))
