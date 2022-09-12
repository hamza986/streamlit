# Import libraries
from ast import Param
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# app ki heading
st.write('''
# Explore Diffrent ML model and datasets
Dekhty han kon sa best hy in main sy?
''')

# data set k naam 1 box main likh k sidebar pay lga do
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Iris", "Breast Cancer", "Wine")
)

# or us k nichy classifier k name ek dabay main dal do
classfier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "RandomForest")
)

# ek function banana ha jis main data set load ho jay
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

# ab es function ko call kr lein aur x,y variable k equal rakhein
x,y = get_dataset(dataset_name)

# ab hm apny dataset ki shape ko print kr lein gy
st.write("Shape of dataset", x.shape)
st.write("Number of classes", len(np.unique(y)))

# next ham different classifier k parameter ko user input main add kr dein gy
def add_parameter_ui(classfier_name):
    params = dict()  # create an empty dictionery
    if classfier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C   # its the degree of correct classification
    elif classfier_name == "KNN":
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K  # its the number of nearest neighbours
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth     # Depth of every tree that grow in Random forest
        n_estimators= st.sidebar.slider('n_estimators', 1,100)
        params['n_estimators'] = n_estimators  # number of trees
    return params

# ab es function ko call kr lein aur params variable k equal rakhein
params = add_parameter_ui(classfier_name)

# ab ham classifier bnayein gy based on classifer_name and params
def get_classifier(classfier_name, params):
    clf = None
    if classfier_name == "SVM":
        clf = SVC(C=params["C"])
    elif classfier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth=params['max_depth'], random_state=1234)
    return clf

# ab es function ko call kr lein aur clf variable k equal rakhein
clf = get_classifier(classfier_name, params)

# ab ham apny dataset ko test aur train main split kr lety han by 80/20 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# ab ham ny apny classifer ki training krni ha
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# model ka ccuracy check kr lein aur app par predict kr dena ha
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classfier_name}")
st.write(f"Accuracy = {acc}")


### plot dataset
# ab ham apny sary features ko 2 dimenional plot main draw kr lein gy using pcs
pca = PCA(2)
x_project = pca.fit_transform(x)

# ab ham apna data 0 or 1 main slice jr dein gy
x1 = x_project[:, 0]
x2 = x_project[:, 1]

fig  = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# plt.show()
st.pyplot(fig)