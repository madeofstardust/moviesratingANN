# Multiple Linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('movie_metadata.csv')
dataset = dataset[dataset['director_name'].notnull()]
dataset = dataset[dataset['country'].notnull()]
X = dataset.iloc[:, [1, 2, 3, 7, 24, 22, 23, 20]].values
y = dataset.iloc[:, 25].values


#getting rid of NaN:
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)

for x in range (1, 7):	
	imputer = imputer.fit((X[:, x].reshape(-1,1)))
	X[:, x] = (imputer.transform(X[:, x].reshape(-1,1))).reshape(1, -1)


df = pd.DataFrame(X)
#encoding categorized data:
from sklearn.preprocessing import LabelEncoder
labelEncoderX1 = LabelEncoder();
X[:, 0] = labelEncoderX1.fit_transform(X[:, 0])

labelEncoderX2 = LabelEncoder();
X[:, 7] = labelEncoderX2.fit_transform(X[:, 7])

df_2 = pd.DataFrame(X)

df_X = pd.DataFrame(X)
df_Y = pd.DataFrame(y)

#dummy vars

from pandas import get_dummies
X_director_dummies = X[:, 0].copy()
X_director_dummies = pd.get_dummies(X_director_dummies, drop_first = True, columns = '0', prefix = 'director')

X_countries_dummies = X[:, 6].copy()
X_countries_dummies = pd.get_dummies(X_countries_dummies, drop_first = True, columns = '0', prefix = 'country')


df_X = pd.DataFrame(X)

del df_X[0]
del df_X[7]

df_3 = df_X


df_X_dir = pd.DataFrame(X_director_dummies)
df_X_cou = pd.DataFrame(X_countries_dummies)

frames = [df_X_dir, df_X, df_X_cou]

X = pd.concat(frames, axis = 1)

# Here i stopped



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

##feature scaling:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting multiple linear regressio into the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting:
y_pred = regressor.predict(X_test)

df2 = pd.DataFrame(X_test)
df3 = pd.DataFrame(X_train)
