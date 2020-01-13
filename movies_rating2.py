# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('movie_metadata.csv')

# Deleting the rows with nulls
dataset = dataset[dataset['director_name'].notnull()]
dataset = dataset[dataset['actor_2_name'].notnull()]
dataset = dataset[dataset['actor_1_name'].notnull()]
dataset = dataset[dataset['actor_3_name'].notnull()]
dataset = dataset[dataset['country'].notnull()]

#Choosing the right rows:
X = dataset.iloc[:1000, [1, 10, 6, 20, 23, 13, 22] ].values
y = dataset.iloc[:1000, 25].values

del dataset

#X = dataset.iloc[:, [1 (dir name), 10( first actor), 6(sec antocr), 20(country), 23(yeR) 13 (cast likes)   22(budget)] ].values


#getting rid of NaN:
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

for x in range (4, 7):	
	imputer = imputer.fit((X[:, x].reshape(-1,1)))
	X[:, x] = (imputer.transform(X[:, x].reshape(-1,1))).reshape(1, -1)

del x

df = pd.DataFrame(X)

#encoding categorized data:
from sklearn.preprocessing import LabelEncoder
labelEncoderX0 = LabelEncoder();
X[:, 0] = labelEncoderX0.fit_transform(X[:, 0])

labelEncoderX1 = LabelEncoder();
X[:, 1] = labelEncoderX1.fit_transform(X[:, 1])

labelEncoderX2 = LabelEncoder();
X[:, 2] = labelEncoderX2.fit_transform(X[:, 2])

labelEncoderX3 = LabelEncoder();
X[:, 3] = labelEncoderX3.fit_transform(X[:, 3])


#dummy vars

from pandas import get_dummies
X_director_dummies = X[:, 0].copy()
X_director_dummies = pd.get_dummies(X_director_dummies, columns = '0', prefix = 'director')

X_1_actor_dummies = X[:, 1].copy()
X_1_actor_dummies = pd.get_dummies(X_1_actor_dummies, columns = '0', prefix = 'actor_1')

X_2_actor_dummies = X[:, 2].copy()
X_2_actor_dummies = pd.get_dummies(X_2_actor_dummies, columns = '0', prefix = 'actor_2')


X_countries_dummies = X[:, 3].copy()
X_countries_dummies = pd.get_dummies(X_countries_dummies, columns = '0', prefix = 'country')

X_year_dummies = X[:, 4].copy()
X_year_dummies = pd.get_dummies(X_year_dummies, columns = '0', prefix = 'year')


df_X = pd.DataFrame(X)

for i in range(0, 5):
	del df_X[i]
	
del i	

X_director_dummies = pd.DataFrame(X_director_dummies)
X_1_actor_dummies = pd.DataFrame(X_1_actor_dummies)
X_2_actor_dummies = pd.DataFrame(X_2_actor_dummies)
X_countries_dummies = pd.DataFrame(X_countries_dummies)
X_year_dummies = pd.DataFrame(X_year_dummies)

frames = [X_director_dummies, X_1_actor_dummies, X_2_actor_dummies, X_countries_dummies, X_year_dummies, df_X]

#Creating the final set:
X = pd.concat(frames, axis = 1)

#getting rid of unnecessary vars:
del X_director_dummies
del X_1_actor_dummies
del X_2_actor_dummies
del X_countries_dummies
del X_year_dummies
del df_X
del frames


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

##feature scaling:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# Fitting multiple linear regression into the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting:
y_pred = regressor.predict(X_test)

## y_pred should more or less equal to y_test - but it does not ;C
