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
#X = dataset.iloc[:, [1, 10, 6, 20, 23, 13, 22] ].values
X = dataset.iloc[:, [1, 10, 6, 20, 13, 23, 22] ].values

y = dataset.iloc[:, 25].values

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

#frames = [X_director_dummies, X_1_actor_dummies, X_2_actor_dummies, X_countries_dummies, X_year_dummies, df_X]

frames = [X_director_dummies, X_1_actor_dummies, X_countries_dummies, df_X]

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

##############################
##end of data preprocessing
##############################


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

##feature scaling:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##ANN!
#Importing Keras:
import keras
#To initialize NN:
from keras.models import Sequential
#to create layesr:
from keras.layers import Dense

##initialize NN (as a sequence of layers):
classifier = Sequential()

##adind the input & first hidden layer
classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu', input_dim =4482))

##second hidden layer:
classifier.add(Dense(output_dim = 20, init = 'uniform', activation = 'relu'))

#Output layerL
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'relu'))

#Compiling
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'] )

##Fitting  ANN to the training set:
## epoch = 1000
##  loss value = 0.02 classifier.fit(X_train, y_train, epochs = 1000)
## dim 50, 5, 1; loss value = 0.018 , error = 2.22 
##classifier.fit(X_train, y_train, epochs = 1000)

##dim 2000, 50, 1  error 2.31
##value = 0.05 lassifier.fit(X_train, y_train, epochs = 100, batch_size = 100)
##dim 100, 10, 1
##value = 2.27 classifier.fit(X_train, y_train, epochs = 800)
##epochs:1000
##: without field "13": dim 100, 10, 1; error value = 1.93
## with field 13 error = 3:
##: without field "13": dim 200, 10, 1; error value = 2.55
##: without field "13": dim 100, 20, 1; error value = 1.69
##: without field "13": dim 100, 30, 1; error value = 1.94
##: without field "13": dim 100, 25, 1  error value = 3.44 (batchsize = 100)
##: with field "13": dim 100, 25, 1  error value = 2.06
##: without field "13": dim 100, 15, 1  error value = 2.32 
##: without field "13": dim 2000, 100, 1  error value = 2.17 
##: without field "13": dim 100, 20, 1; error value = 2.7
##: withfield "13": dim 100, 20, 1; error value = 1.9
##: withfield "13": dim 100, 30, 1; error value = 1.75
##: withfield "13": dim 100, 40, 1; error value = 1.9
##: withfield "13": dim 100, 50, 1; error value = 1.98
##: withfield "13": dim 200, 60, 1; error value = 2.16
##: withfield "13": dim 100, 20, 1; error value = 1.88

##the most optimal : 100,20,1 and epochs = 1000

classifier.fit(X_train, y_train, epochs = 1000)
#predictions:
y_pred = classifier.predict(X_test)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_pred, y_test)
