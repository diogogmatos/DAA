import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from pandas import DataFrame

# Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier(max_depth=2, random_state=2024)
    print("Fitting Decision Tree model...")
    dt_model.fit(X_train, y_train)

    dt_score = dt_model.score(X_test, y_test)
    print("Decision Tree Accuracy: %.2f%%" % (dt_score * 100))

    return dt_model

# Support Vector Machine
def svm(X_train, y_train, X_test, y_test):
    svm_model = SVC(random_state=2024)
    print("Fitting SVM model...")
    svm_model.fit(X_train, y_train)

    svm_score = svm_model.score(X_test, y_test)
    print("SVM Accuracy: %.2f%%" % (svm_score * 100))

    return svm_model

# Bagging
def bagging(X_train, y_train, X_test, y_test, dt_model):
    sss = StratifiedShuffleSplit(test_size=20, random_state=2024)
    bagging_model = BaggingClassifier(
        estimator=dt_model, random_state=2024)

    n_estimators = [10, 40, 60, 80, 100, 160]
    parameters = {'n_estimators': n_estimators}
    print("Grid Searching Bagging model...")
    grid_bg = GridSearchCV(estimator=bagging_model, param_grid=parameters, cv=sss)
    grid_bg.fit(X_train, y_train)

    bst_bg_model = grid_bg.best_estimator_
    print("Fitting Bagging model...")
    bst_bg_model.fit(X_train, y_train)

    bg_score = bst_bg_model.score(X_test, y_test)
    print("Bagging Accuracy: %.2f%%" % (bg_score * 100))

    return bst_bg_model

# Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=2024)
    print("Fitting Random Forest model...")
    rf_model.fit(X_train, y_train)

    rf_score = rf_model.score(X_test, y_test)
    print("Random Forest Accuracy: %.2f%%" % (rf_score * 100))

    return rf_model

# Gradient Boosting
def gradient_boosting(X_train, y_train, X_test, y_test):
    gbc_model = GradientBoostingClassifier(random_state=2024)
    print("Fitting Gradient Boosting model...")
    gbc_model.fit(X_train, y_train)

    gbc_score = gbc_model.score(X_test, y_test)
    print("Gradient Boosting Accuracy: %.2f%%" % (gbc_score * 100))

    return gbc_model

# XGBoost
def xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(max_depth=1, objective='reg:squarederror')
    
    # transform string labels into integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    print("Fitting XGBoost model...")
    xgb_model.fit(X_train, y_train)

    xgb_score = xgb_model.score(X_test, y_test)
    print("XGBoost Accuracy: %.2f%%" % (xgb_score * 100))

    return xgb_model, le

# Stacking Classifier
def stacking(X_train, y_train, X_test, y_test, dt_model, rf_model, gbc_model):
    estimators = [("dt", dt_model), ("rf", rf_model), ("gbc", gbc_model)]
    st_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    print("Fitting Stacking model...")
    st_model.fit(X_train, y_train)

    st_score = st_model.score(X_test, y_test)
    print("Stacking Accuracy: %.2f%%" % (st_score * 100))

    return st_model