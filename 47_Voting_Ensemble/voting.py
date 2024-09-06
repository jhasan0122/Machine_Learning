
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.ensemble import VotingClassifier

@st.cache_data
def load_data():
    X, y = make_blobs(n_features=2, centers=2, random_state=6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

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
    plt.title(f'Decision Boundary: {model.__class__.__name__}')
    st.pyplot(plt)
    plt.clf()  # Clear the figure after plotting

def com(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')

    # Find convex hull
    points = np.c_[xx.ravel(), yy.ravel()]
    hull = ConvexHull(points[Z.ravel() == 1])  # Assuming class 1 is the positive class
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Convex Hull and Decision Boundary')
    st.pyplot(plt)
    plt.clf()

st.title('Logistic Regressor Hyperparameter Tuning')
st.markdown('Logistic Regressor Hyperparameter Tuning')

with st.form('train_model'):
    selected_estimators = st.multiselect(
        'Estimator',
        options=['KNN', 'Logistic Regression', 'Gaussian Naive Bayes', "SVM", 'Random Forest']
    )

    submitted = st.form_submit_button("Train")

    estimators = []

    if submitted:
        for estimator in selected_estimators:
            if estimator == 'KNN':
                clf1 = KNeighborsClassifier()
                estimators.append(('KNN', clf1))

            elif estimator == 'Logistic Regression':
                clf2 = LogisticRegression()
                estimators.append(('Logistic Regression', clf2))

            elif estimator == 'Gaussian Naive Bayes':
                clf3 = GaussianNB()
                estimators.append(('Gaussian Naive Bayes', clf3))

            elif estimator == 'SVM':
                clf4 = SVC(probability=True)
                estimators.append(('SVM', clf4))

            elif estimator == 'Random Forest':
                clf5 = RandomForestClassifier()
                estimators.append(('Random Forest', clf5))

        eclf = VotingClassifier(estimators=estimators, voting='soft')

        # Train the ensemble
        eclf.fit(X_train, y_train)
        plot_decision_boundary(X_train, y_train, eclf)

        for name, model in estimators:
            model.fit(X_train, y_train)  # Train the model
            scores = cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy')
            st.metric(name, str(100 * np.round(np.mean(scores), 2)) + ' %')
            plot_decision_boundary(X_train, y_train, model)  # Plot decision boundary
