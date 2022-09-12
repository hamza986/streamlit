from statistics import mode
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# making containers
headers = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with headers:
    st.title("Titanic App")
    st.text("This is a simple app to predict the survival of the titanic passengers")

with data_sets:
    st.header("Data Sets")
    st.text("This is the data set used for the prediction")

    #import data
    df = sns.load_dataset("titanic")
    df = df.dropna()
    st.write(df.head(10))

    st.subheader("Data Description")
    st.bar_chart(df['sex'].value_counts())

    # other barplot
    st.subheader('Class k hisab sy faraq')
    st.bar_chart(df['class'].value_counts())

    # barchart
    st.bar_chart(df['age'].sample(10))

with features:
    st.header("Features")
    st.text("This is the features used for the prediction")
    st.markdown("1. **Feature 1** : This is the first feature")
    st.markdown("2. **Feature 2** : This is the second feature")
    st.markdown("3. **Feature 3** : This is the third feature")


with model_training:
    st.header("Model Training")
    st.text("This is the model used for the prediction")

    # making columns
    input, display = st.columns(2)

    # selction points in 1st column
    max_depth = input.slider("How many people do you know?", min_value=10, max_value=100, value=20, step=5)

    # display points in 2nd column
    n_estimators = input.selectbox("How many trees should be there in Rf?", options=[50, 60, 70, 100, 'No limit'])

    # adding list of features
    input.write(df.columns)

# input_features
input_features = input.text_input("Enter the features")

# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# apply a condition here for NO Limit
if n_estimators == 'No limit':
    model = RandomForestRegressor(max_depth=max_depth)
else:
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# deine X and y
X  = df[[input_features]]
y = df[['fare']]

# fit the model
model.fit(X, y)

# prediction
pred = model.predict(y)

# display the metrices
display.subheader('Mean Absolute error of model is : ')
display.write(mean_absolute_error(y, pred))
display.subheader('Mean Squared error of model is : ')
display.write(mean_squared_error(y, pred))
display.subheader('R squared score of model is : ')
display.write(r2_score(y, pred))