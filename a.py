import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler.fit(X_train)
sns.set_palette('husl')
# matplotlib inline
# config InlineBackend.figure_format = 'retina'
#load the dataset
dataset = pd.read_csv(r'C:\Users\aforb\Downloads\iris_dataset.csv')
print(dataset)
label_mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}
dataset_X = dataset.drop(['Species'], axis=1).values
dataset_Y = dataset.Species.replace(label_mapping).values.reshape(dataset.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.3, shuffle = True, random_state = 123)
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train.ravel())
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
