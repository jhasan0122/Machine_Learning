#run: streamlit run 8_logistic_regression_hyperparameter_tuning.py
import streamlit as st

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score

from sklearn.datasets import make_blobs


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


@st.cache_data
def load_data():
    X, y = make_blobs(n_features=2,centers=2,random_state=6)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


def train_model(penalty,C,solver,max_iter,multi_class,l1_ratio):
    lr = LogisticRegression(penalty=penalty,C=C,solver=solver,max_iter=max_iter,multi_class=multi_class,l1_ratio=l1_ratio)
    lr.fit(X_train, y_train)
    return lr

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    st.pyplot(plt)


# Dashboard
st.title('Logistic Regressor Hyperparameter Tuning')
st.markdown('Logistic Regressor Hyperparameter Tuning')

with (st.form('train_model')):
    col1, col2 = st.columns(2, gap='large')

    with col1:
        penalty = st.selectbox('Penalty',options=['l1', 'l2', 'elasticnet', None],index=1)
        C = st.number_input("C",value=1.0)
        solver = st.selectbox("Solver",options=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],index=0)

    with col2:
        max_iter = st.number_input("max_iter", value=100, step=1, format="%d")
        multi_class = st.selectbox("multi_class", options=['auto', 'ovr', 'multinomial'], index=0)
        l1_ratio = st.number_input("l1_ratio", value=None)

    submitted = st.form_submit_button("Train")

    if submitted:
        lr = train_model(penalty,C,solver,max_iter,multi_class,l1_ratio)

        y_test_pred = lr.predict(X_test)
        y_train_pred = lr.predict(X_train)
        st.metric('Test Accuracy', value="{:.2f} %".format(100 * accuracy_score(y_test, y_test_pred)))

        plot_decision_boundary(X_train, y_train, lr)




