# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from easygui import *

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
while(1):

	msg99 = "Select Type Of Classification"
	choice99 = ["logistic training","Logistic testing","K-NN training","K-NN testing","Naive-Bayes training","Naive-Bayes testing","Quit"]
	reply99 = buttonbox(msg99, choices=choice99,image = "image2.jpg")
	if reply99 == "logistic training":
		# Fitting Logistic Regression to the Training set
		from sklearn.linear_model import LogisticRegression
		classifier = LogisticRegression(random_state = 0)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)

		# Visualising the Training set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_train, y_train
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('Logistic Regression (Training set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()


	if reply99 == "Logistic testing":
		# Fitting Logistic Regression to the Training set
		from sklearn.linear_model import LogisticRegression
		classifier = LogisticRegression(random_state = 0)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		print(cm)
		print(cm[0][0]+cm[1][1])
		
		# Visualising the Test set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_test, y_test
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('Logistic Regression (Test set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()
		msgbox("Confusion matrix is: \n\n"+str(cm)+"\n\n\nAccuracy of Logistic Regression is: " + str(cm[0][0]+cm[1][1])+"%")
		
		
	if reply99 == "K-NN training":
		from sklearn.neighbors import KNeighborsClassifier
		classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)

		# Visualising the Training set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_train, y_train
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('K-NN (Training set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()
		
	if reply99 == "K-NN testing":
		from sklearn.neighbors import KNeighborsClassifier
		classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		print(cm[0][0]+cm[1][1])
		# Visualising the Training set results
		# Visualising the Test set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_test, y_test
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('K-NN(Test set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()
		
		msgbox("Confusion matrix is: \n\n"+str(cm)+"\n\n\nAccuracy of KNN is: " + str(cm[0][0]+cm[1][1])+"%")




	if reply99 == "Naive-Bayes training":
		from sklearn.naive_bayes import GaussianNB
		classifier = GaussianNB()
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		# Visualising the Training set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_train, y_train
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('Naive Bayes (Training set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()
		
	if reply99 == "Naive-Bayes testing":
		from sklearn.naive_bayes import GaussianNB
		classifier = GaussianNB()
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_test)
		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(y_test, y_pred)
		print(cm[0][0]+cm[1][1])
		# Visualising the Training set results
		# Visualising the Test set results
		from matplotlib.colors import ListedColormap
		X_set, y_set = X_test, y_test
		X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
				     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
		plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
		plt.xlim(X1.min(), X1.max())
		plt.ylim(X2.min(), X2.max())
		for i, j in enumerate(np.unique(y_set)):
		    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
				c = ListedColormap(('red', 'green'))(i), label = j)
		plt.title('Naive Bayes(Test set)')
		plt.xlabel('Age')
		plt.ylabel('Estimated Salary')
		plt.legend()
		plt.show()
		
		msgbox("Confusion matrix is: \n\n"+str(cm)+"\n\n\nAccuracy of Naive Bayes is: " + str(cm[0][0]+cm[1][1])+"%")
		
	if reply99 == "Quit":
		sys.exit()	
