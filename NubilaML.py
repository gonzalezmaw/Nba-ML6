import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from SLR import SLR
from RFR import RFR
from RFC import RFC
from PRM import PRM
from MRFR import MRFR
from MLPR import MLPR
from MLPC import MLPC
from ClusteringModels import ClusteringModels
from QuickClassifiers import QuickClassifiers
from QuickRegressors import QuickRegressors
from ClassifierModels import ClassifierModels
from RegressorModels import RegressorModels

#from Pyspark import Pyspark
#from RegressorsPyspark import RegressorsPyspark
#from ClassifiersPyspark import ClassifiersPyspark
#from ClusteringPyspark import ClusteringPyspark
from TimeSeries import TimeSeries
from AnomalyDetection import AnomalyDetection

from PIL import Image

img = Image.open('NubilaIcono.png')

st.set_page_config(page_title="Nubyla", page_icon=img)
st.title("Machine Learning for Everyone")

st.sidebar.write("""
# **nubyla**

""")

# options = st.multiselect(
# 'What are your favorite colors',['Green', 'Yellow', 'Red', 'Blue'],['Yellow', 'Red'])
#st.write('You selected:', options)


CategorySelect_name = st.sidebar.selectbox(
    "Select a category", ("Regression", "Classification", "Clustering", "Ranking", "Time Series", "Anomaly Detection"))

if CategorySelect_name == "Regression":
    OptimizeBigData = st.sidebar.checkbox('Optimize for Big Data')

elif CategorySelect_name == "Classification":
    OptimizeBigData = st.sidebar.checkbox('Optimize for Big Data')

elif CategorySelect_name == "Clustering":
    OptimizeBigData = st.sidebar.checkbox('Optimize for Big Data')
elif CategorySelect_name == "Ranking":
    OptimizeBigData = st.sidebar.checkbox('Optimize for Big Data')
else:
    OptimizeBigData = False


if OptimizeBigData == False and CategorySelect_name == 'Regression':

    if CategorySelect_name == "Regression":
        modelSelect_name = ""

        modelSelectSubcategory_name = st.sidebar.selectbox(
            "Select a subcategory", ("Single Variable Regression", "Multiple Variable Regression"))

        if modelSelectSubcategory_name == "Single Variable Regression":
            modelSelect_name = st.sidebar.selectbox(
                "Select an algorithm", ("Simple Linear Regression", "Polynomial Regression", "Random Forest Regressor"))

        elif modelSelectSubcategory_name == "Multiple Variable Regression":

            modelSelect_name = st.sidebar.selectbox("Select an algorithm", ("Multiple Linear Regressor", "Support Vector Machines Regressor", "Bayesian Ridge Regressor",
                                                                            "Decision Tree Regressor", "Extra Trees Regressor", "Random Forest Regressor", "K-Nearest Neighbors Regressor", "Gradient Boosting Regressor", "Extreme Gradient Boosting Regressor", "Gaussian Process Regressor", "Stochastic Gradient Descent Regressor",
                                                                            "Light Gradient Boosting Machine Regressor", "CatBoost Regressor", "AdaBoost Regressor", "Bagging Regressor", "Passive Aggressive Regressor",
                                                                            "Elastic Net Regressor", "Lasso Regressor", "Ridge Regressor", "Huber Regressor", "Kernel Ridge Regressor",
                                                                            "Tweedie Regressor", "TheilSen Regressor", "Orthogonal Matching Pursuit Regressor", "Histogram Gradient Boosting Regressor", "Least Angle Regressor",
                                                                            "Lasso Least Angle Regressor", "Automatic Relevance Determination Regressor", "Random Sample Consensus Regressor", "Perceptron Regressor", "Natural Gradient Boosting Regressor", "Neural Network Regression"))

elif OptimizeBigData == False and CategorySelect_name == 'Classification':
    if CategorySelect_name == "Classification":
        modelSelect_name = st.sidebar.selectbox("Select an algorithm", ("Random Forest Classifier", "Support Vector Machines Classifier", "Logistic Regression Classifier", "Naive Bayes Classifier", "Decision Tree Classifier", "Extra Trees Classifier",
                                                                        "K-Nearest Neighbors Classifier", "Gradient Boosting Classifier", "Extreme Gradient Boosting Classifier", "Gaussian Process Classifier", "Stochastic Gradient Descent Classifier",
                                                                        "Light Gradient Boosting Machine Classifier", "CatBoost Classifier", "AdaBoost Classifier", "Bagging Classifier", "Passive Aggressive Classifier",
                                                                        "Linear Discriminant Analysis Classifier", "Quadratic Discriminant Analysis Classifier", "Linear Support Vector Machine Classifier", "Ridge Classifier", "Natural Gradient Boosting Classifier",
                                                                        "Neural Network Classification"))

elif OptimizeBigData == False and CategorySelect_name == 'Clustering':
    if CategorySelect_name == "Clustering":
        modelSelect_name = st.sidebar.selectbox(
            "Select an algorithm", ("K-Means Clustering", "Hierarchical Clustering", "Spectral Clustering"))

elif CategorySelect_name == "Ranking":
    modelSelect_name = st.sidebar.selectbox(
        "Select a method", ("Quick Comparison Regressors", "Quick Comparison Classifiers"))

elif CategorySelect_name == "Bigdata Analysis":
    modelSelect_name = "Bigdata Analysis"

elif CategorySelect_name == "Time Series":
    modelSelect_name = "Time Series"

elif CategorySelect_name == "Anomaly Detection":
    modelSelect_name = "Anomaly Detection"


# modelSelect_name = st.sidebar.selectbox(
#    "Select a Model", ("Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regressor", "Support Vector Machines Regressor", "Bayesian Ridge Regressor",
#                       "Decision Tree Regressor", "Extra Trees Regressor", "Random Forest Regressor", "K-Nearest Neighbors Regressor", "Gradient Boosting Regressor", "Extreme Gradient Boosting Regressor", "Gaussian Process Regressor", "Stochastic Gradient Descent Regressor",
#                       "Light Gradient Boosting Machine Regressor", "CatBoost Regressor", "AdaBoost Regressor", "Bagging Regressor", "Passive Aggressive Regressor",
#                       "Elastic Net Regressor", "Lasso Regressor", "Ridge Regressor", "Huber Regressor", "Kernel Ridge Regressor",
#                       "Tweedie Regressor", "TheilSen Regressor", "Orthogonal Matching Pursuit Regressor", "Histogram Gradient Boosting Regressor", "Least Angle Regressor",
#                       "Lasso Least Angle Regressor", "Automatic Relevance Determination Regressor", "Random Sample Consensus Regressor", "Perceptron Regressor", "Natural Gradient Boosting Regressor",
#                       "Random Forest Classifier", "Support Vector Machines Classifier", "Logistic Regression Classifier", "Naive Bayes Classifier", "Decision Tree Classifier", "Extra Trees Classifier",
#                       "K-Nearest Neighbors Classifier", "Gradient Boosting Classifier", "Extreme Gradient Boosting Classifier", "Gaussian Process Classifier", "Stochastic Gradient Descent Classifier",
#                       "Light Gradient Boosting Machine Classifier", "CatBoost Classifier", "AdaBoost Classifier", "Bagging Classifier", "Passive Aggressive Classifier",
#                       "Linear Discriminant Analysis Classifier", "Quadratic Discriminant Analysis Classifier", "Linear Support Vector Machine Classifier", "Ridge Classifier", "Natural Gradient Boosting Classifier",
#                       "Quick Comparison Regressors", "Quick Comparison Classifiers",
#                       "NN - Multi-Layer Perceptron",
#                       "K-Means Clustering", "Hierarchical Clustering", "Spectral Clustering",
#                       "Bigdata Analysis", "Time Series", "Anomaly Detection"))


if OptimizeBigData == False and CategorySelect_name == 'Regression':
    if modelSelect_name == "Simple Linear Regression":
        st.write("""
            ## **Simple Linear Regression**
            """)
        st.write("""
        ### **Simple Regression Method**
        """)
        SLR()

    elif modelSelect_name == "Multiple Linear Regressor":
        st.write("""
            ## **Multiple Linear Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Support Vector Machines Regressor":
        st.write("""
            ## **Support Vector Machines Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Bayesian Ridge Regressor":
        st.write("""
            ## **Bayesian Ridge Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Decision Tree Regressor":
        st.write("""
            ## **Decision Tree Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Extra Trees Regressor":
        st.write("""
            ## **Extra Trees Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Random Forest Regressor":
        st.write("""
            ## **Random Forest Regressor**
            """)
        if modelSelectSubcategory_name == "Single Variable Regression":
            st.write("""
                    ### **Simple Regression Method**
                    """)
            RFR()

        elif modelSelectSubcategory_name == "Multiple Variable Regression":
            st.write("""
                    ### **Multiple Regression Method**
                    """)
            RegressorModels(modelSelectSubcategory_name)

    elif modelSelect_name == "Gradient Boosting Regressor":
        st.write("""
            ## **Gradient Boosting Regressor**
            """)
        st.write("""
        ### **Multivariate Regression Method**
        """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Extreme Gradient Boosting Regressor":
        st.write("""
            ## **Extreme Gradient Boosting Regressor**
            """)
        st.write("""
        ### **Multivariate Regression Method**
        """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Gaussian Process Regressor":
        st.write("""
            ## **Gaussian Process Regressor**
            """)
        st.write("""
        ### **Multivariate Regression Method**
        """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Stochastic Gradient Descent Regressor":
        st.write("""
            ## **Stochastic Gradient Descent Regressor**
            """)
        st.write("""
        ### **Multivariate Regression Method**
        """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "K-Nearest Neighbors Regressor":
        st.write("""
            ## **K-Nearest Neighbors Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Light Gradient Boosting Machine Regressor":
        st.write("""
            ## **Light Gradient Boosting Machine Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "CatBoost Regressor":
        st.write("""
            ## **CatBoost Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "AdaBoost Regressor":
        st.write("""
            ## **AdaBoost Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Bagging Regressor":
        st.write("""
            ## **Bagging Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Passive Aggressive Regressor":
        st.write("""
            ## **Passive Aggressive Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Elastic Net Regressor":
        st.write("""
            ## **Elastic Net Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Lasso Regressor":
        st.write("""
            ## **Lasso Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Ridge Regressor":
        st.write("""
            ## **Ridge Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Huber Regressor":
        st.write("""
            ## **Huber Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Kernel Ridge Regressor":
        st.write("""
            ## **Kernel Ridge Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Tweedie Regressor":
        st.write("""
            ## **Tweedie Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "TheilSen Regressor":
        st.write("""
            ## **TheilSen Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Orthogonal Matching Pursuit Regressor":
        st.write("""
            ## **Orthogonal Matching Pursuit Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Histogram Gradient Boosting Regressor":
        st.write("""
            ## **Histogram Gradient Boosting Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Least Angle Regressor":
        st.write("""
            ## **Least Angle Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Lasso Least Angle Regressor":
        st.write("""
            ## **Lasso Least Angle Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Automatic Relevance Determination Regressor":
        st.write("""
            ## **Automatic Relevance Determination Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Random Sample Consensus Regressor":
        st.write("""
            ## **Random Sample Consensus Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Perceptron Regressor":
        st.write("""
            ## **Perceptron Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

    elif modelSelect_name == "Natural Gradient Boosting Regressor":
        st.write("""
            ## **Natural Gradient Boosting Regressor**
            """)

        st.write("""
            ### **Multiple Regression Method**
            """)
        RegressorModels(modelSelect_name)

#  CLASSIFICATION CASES
if OptimizeBigData == False and CategorySelect_name == 'Classification':
    if modelSelect_name == "Support Vector Machines Classifier":
        st.write("""
            ## **Support Vector Machines Classifier**
            """)
        st.write("""
        ### **Multivariate Classification Method**
        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Logistic Regression Classifier":
        st.write("""
            ## **Logistic Regression Classifier**
            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Naive Bayes Classifier":
        st.write("""
            ## **Naive Bayes Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Decision Tree Classifier":
        st.write("""
            ## **Decision Tree Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Extra Trees Classifier":
        st.write("""
            ## **Extra Trees Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "K-Nearest Neighbors Classifier":
        st.write("""
            ## **K-Nearest Neighbors Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Random Forest Classifier":
        st.write("""
            ## **Random Forest Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Gradient Boosting Classifier":
        st.write("""
            ## **Gradient Boosting Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Extreme Gradient Boosting Classifier":
        st.write("""
            ## **Extreme Gradient Boosting Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Gaussian Process Classifier":
        st.write("""
            ## **Gaussian Process Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Stochastic Gradient Descent Classifier":
        st.write("""
            ## **Stochastic Gradient Descent Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Light Gradient Boosting Machine Classifier":
        st.write("""
            ## **Light Gradient Boosting Machine Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "CatBoost Classifier":
        st.write("""
            ## **CastBoost Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "AdaBoost Classifier":
        st.write("""
            ## **AdaBoost Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Bagging Classifier":
        st.write("""
            ## **Bagging Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Passive Aggressive Classifier":
        st.write("""
            ## **Passive Aggressive Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Linear Discriminant Analysis Classifier":
        st.write("""
            ## **Linear Discriminant Analysis Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Quadratic Discriminant Analysis Classifier":
        st.write("""
            ## **Quadratic Discriminant Analysis Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Linear Support Vector Machine Classifier":
        st.write("""
            ## **Linear Support Vector Machine Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Ridge Classifier":
        st.write("""
            ## **Ridge Classifier Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

    elif modelSelect_name == "Natural Gradient Boosting Classifier":
        st.write("""
            ## **Natural Gradient Boosting Classifier**

            """)
        st.write("""
        ### **Multivariate Classification Method**

        """)
        ClassifierModels(modelSelect_name)

if OptimizeBigData == False and CategorySelect_name == 'Regression':
    if modelSelect_name == "Polynomial Regression":
        st.write("""
            ## **Polynomial Regression**

            """)
        st.write("""
        ### **Single-Variable Regression Method**

        """)
        PRM()

    # This model is included and more complete in RegressorModels.py
    # and it was left in the code for reference only

    elif modelSelect_name == "Random Forest Regression":
        st.write("""
            ## **Multiple Random Forest Regression**

            """)
        st.write("""
        ### **Multiple Variable Regression Method**

        """)
        MRFR()

    elif modelSelect_name == "Neural Network Regression":

        modelSelect_Type = "Neural Network Regression"
        st.write("""
            ## **Multi-Layer Perceptron Regressor**

            """)

        st.write("""
        ### **Neural Network (supervised) Regression Method**

        """)
        MLPR()

if OptimizeBigData == False and CategorySelect_name == 'Classification':
    if modelSelect_name == "Neural Network Classification":
        modelSelect_Type = "Neural Network Classification"
        st.write("""
            ## **Multi-Layer Perceptron Classification**

            """)
        st.write("""
        ### **Neural Network (supervised) Classification Method**

        """)
        MLPC()

if OptimizeBigData == False and CategorySelect_name == 'Clustering':
    if modelSelect_name == "K-Means Clustering":
        st.write("""
            ## **K-Means Clustering**

            """)

        st.write("""
        ### **Unsupervised Learning Method**

        """)
        ClusteringModels(modelSelect_name)

    elif modelSelect_name == "Hierarchical Clustering":
        st.write("""
            ## **Hierarchical Clustering**

            """)

        st.write("""
        ### **Unsupervised Learning Method**

        """)
        ClusteringModels(modelSelect_name)

    elif modelSelect_name == "Spectral Clustering":
        st.write("""
            ## **Spectral Clustering**

            """)

        st.write("""
        ### **Unsupervised Learning Method**

        """)
        ClusteringModels(modelSelect_name)

if OptimizeBigData == False and CategorySelect_name == 'Ranking':

    if modelSelect_name == "Quick Comparison Classifiers":
        st.write("""
            ## **Quick Comparison Classifiers**

            """)
        QuickClassifiers()

if OptimizeBigData == False and CategorySelect_name == 'Ranking':
    if modelSelect_name == "Quick Comparison Regressors":
        st.write("""
            ## **Quick Comparison Regressors**

            """)
        QuickRegressors()

if OptimizeBigData == True and CategorySelect_name == 'Regression':
    if CategorySelect_name == 'Regression':

        if CategorySelect_name == 'Regression':
            st.write("""
            ## **Bigdata - Regression Analysis**

            """)
            selectModelRegressor = st.sidebar.selectbox("Select an algorithm", ("Linear Regressor", "Generalized Linear Regressor", "Decision Tree Regressor",
                                                                                "Random Forest Regressor", "Gradient-Boosted Tree Regressor"))

            if selectModelRegressor == "Linear Regressor":

                st.write("""
                            ### **Linear Regressor**

                            """)

            elif selectModelRegressor == "Generalized Linear Regressor":
                st.write("""
                            ### **Generalized Linear Regressor**

                            """)

            elif selectModelRegressor == "Decision Tree Regressor":
                st.write("""
                            ### **Decision Tree Regressor**

                            """)

            elif selectModelRegressor == "Random Forest Regressor":
                st.write("""
                            ### **Random Forest Regressor**

                            """)

            elif selectModelRegressor == "Gradient-Boosted Tree Regressor":
                st.write("""
                            ### **Gradient-Boosted Tree Regressor**

                            """)

            #RegressorsPyspark(selectModelRegressor)

if OptimizeBigData == True and CategorySelect_name == 'Classification':

    if CategorySelect_name == 'Classification':
        st.write("""
            ## **Bigdata - Classification Analysis**

            """)
        selectModelClassifier = st.sidebar.selectbox("Select an algorithm", ("Decision Tree Classifier", "Logistic Regression Classifier",
                                                                             "Random Forest Classifier", "Navy Bayes Classifier"))

        if selectModelClassifier == "Decision Tree Classifier":

            st.write("""
                            ### **Decision Tree Classifier**

                            """)

        elif selectModelClassifier == "Logistic Regression Classifier":

            st.write("""
                            ### **Logistic Regression Classifier**

                            """)

        elif selectModelClassifier == "Random Forest Classifier":

            st.write("""
                            ### **Random Forest Classifier**

                            """)

        elif selectModelClassifier == "Navy Bayes Classifier":

            st.write("""
                            ### **Navy Bayes Classifier**

                            """)

        #ClassifiersPyspark(selectModelClassifier)

if OptimizeBigData == True and CategorySelect_name == 'Ranking':

    if modelSelect_name == "Quick Comparison Regressors":

        if modelSelect_name == "Quick Comparison Regressors":
            st.write("""
            ## **Bigdata - Regressor Ranking Analysis**

            """)
            SelectMethod = 'Regressor Ranking'

    elif modelSelect_name == "Quick Comparison Classifiers":
        st.write("""
            ## **Bigdata - Classifier Ranking Analysis**

            """)
        SelectMethod = 'Classifier Ranking'

    #Pyspark(SelectMethod)

if OptimizeBigData == True and CategorySelect_name == 'Clustering':
    if CategorySelect_name == "Clustering":
        st.write("""
            ## **Bigdata - Clustering Analysis**

            """)
        selectModelClustering = st.sidebar.selectbox(
            "Select an algorithm", ("K-Means", "Gaussian Mixture"))

        if selectModelClustering == "K-Means":

            st.write("""
                            ### **K-Means Clustering**

                            """)

        if selectModelClustering == "Gaussian Mixture":

            st.write("""
                            ### **Gaussian Mixture Clustering**

                            """)
        #ClusteringPyspark(selectModelClustering)


if OptimizeBigData == False and CategorySelect_name == 'Time Series':
    if modelSelect_name == "Time Series":
        st.write("""
        ## **Time Series**

        """)
        TimeSeries()

if OptimizeBigData == False and CategorySelect_name == 'Anomaly Detection':
    if modelSelect_name == "Anomaly Detection":
        st.write("""
        ## **Anomaly Detection in Time Series**

        """)
        AnomalyDetection()
