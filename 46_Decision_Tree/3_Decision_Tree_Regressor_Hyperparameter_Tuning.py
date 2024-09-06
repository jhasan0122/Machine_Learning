#run: streamlit run 3_Decision_Tree_Regressor_Hyperparameter_Tuning.py
import streamlit as st

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split


from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_csv('HousingData.csv')

    df = df.dropna()

    X = df.iloc[:, 0:13]
    y = df.iloc[:, 13]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


def train_model(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes,
                min_impurity_decrease):
    dtc = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease)
    dtc.fit(X_train, y_train)
    return dtc


# Dashboard
st.title('Decision Tree Regressor Hyperparameter Tuning')
st.markdown('Decision Tree Regressor Hyperparameter Tuning')

with st.form('train_model'):
    col1, col2 = st.columns(2, gap='large')

    with col1:
        criterion = st.selectbox("Criterion",index=0, options=['squared_error', 'friedman_mse', 'absolute_error','poisson'])
        splitter = st.selectbox("Splitter",index=0, options=['best', 'random'])
        max_depth = st.number_input("Max Depth",min_value=1, max_value=100, step=1, format="%d",value=None)
        min_samples_split = st.slider("min_samples_split", min_value=2, max_value=150,value=2)
        min_samples_leaf = st.slider("min_samples_leaf", min_value=1, max_value=150,value=1)

    with col2:
        max_leaf_nodes = st.number_input("max_leaf_nodes",min_value=2, step=1, format="%d",value=None)
        min_impurity_decrease = st.number_input("min_impurity_decrease",value=0.0)

    submitted = st.form_submit_button("Train")

    if submitted:
        dtr = train_model(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes,min_impurity_decrease)

        y_test_pred = dtr.predict(X_test)
        y_train_pred = dtr.predict(X_train)
        st.metric('Test Accuracy', value="{:.2f} %".format(100 * r2_score(y_test, y_test_pred)))

        fig, ax = plt.subplots(figsize=(20,10))
        plot_tree(dtr, filled=True,feature_names=X_train.columns,ax=ax)
        st.pyplot(fig)
