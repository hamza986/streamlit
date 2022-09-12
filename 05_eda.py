import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# web app title
st.markdown('''
# **Exploratory Data Analysis App**
This app is developed by Ameer Hamza called **EDA App** ''')

# How to upload a file from pc

with st.sidebar.header('1. Upload your (CSV) data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    df  = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

# profiling report for pandas
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st_profile_report(pr)
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    st.header('**Profiling Report with Pandas**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Example data
        @st.cache
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5), columns=['aslam','bandook','chaku', 'danda', 'enam'])
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DF**')
        st.write(df)
        st.write('---')
        st.header('**Profiling Report with Pandas**')
        st_profile_report(pr)
