# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('pulsar_stars.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/5, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

###############################
#X = sc.transform(X)
#y_pred1 = classifier.predict(X)
#y_pred = y_pred1
#y_test = y
###############################

from sklearn.metrics import classification_report
target_names = ['0', '1']
print('\nConfusion Report')
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_test, y_pred)
print('\nError')
print(error)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix')
print(cm)

plt.imshow(cm, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(target_names))
_ = plt.xticks(tick_marks, target_names)
_ = plt.yticks(tick_marks, target_names)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset[['target_class']]], axis = 1)
#print(finalDf)

fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g']
target_names = [0, 1]
for target, color in zip(target_names,colors):
    indicesToKeep = finalDf['target_class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 1)
ax.legend(target_names)
ax.grid()