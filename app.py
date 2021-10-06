from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st 
import numpy as np
st.title("IRIS FLOWER CLASSIFICATION")
var = load_iris() 
x = var.data     #input
y = var.target   #output 
model = KNeighborsClassifier(n_neighbors = 13)
model.fit(x,y)
xmin = np.min(x, axis = 0) #min values of inputs to take all the features 
xmax = np.max(x, axis = 0) #max values of inputs to take all the features 
sepal_length = st.slider('Sepal Length',float(xmin[0]),float(xmax[0]))
sepal_width =  st.slider('Sepal Width',float(xmin[1]),float(xmax[1]))
petal_length = st.slider('Petal Length',float(xmin[2]),float(xmax[2]))
petal_width =  st.slider('Petal width',float(xmin[3]),float(xmax[3]))

y_pred = model.predict([[sepal_length , sepal_width, petal_length, petal_width]])
op = ['Iris-Setosa','Iris-Versicolor','Iris-Virginica']
st.title(op[y_pred[0]])  #to obtainvalues in 1 dimension, we are giving y_pred[0]
