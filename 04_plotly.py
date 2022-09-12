from turtle import width
import streamlit as st
import plotly.express as px
import pandas as pd


# Load data
st.title("Plotly and Streamlit plot Example")
df = px.data.gapminder()
st.write(df)
# st.write(df.head())
st.write(df.columns)

# summary stat
st.write(df.describe())

# data management
year_option = df['year'].unique().tolist()
year = st.selectbox("Select year", year_option, 0)


#df = df[df['year'] == year]

# plotting 
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent", hover_name="continent",
                 log_x=True, size_max=60, range_x=[100,100000], range_y=[25,90],
                 animation_frame="year", animation_group="country"
                 )
fig.update_layout(width=1000, height= 600)
st.write(fig)