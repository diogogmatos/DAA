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
from sklearn.metrics import f1_score

# Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=987654321)
    print("Fitting Decision Tree model...")
    dt_model.fit(X_train, y_train)

    dt_score = dt_model.score(X_test, y_test)
    print("Decision Tree Accuracy: %.2f%%" % (dt_score * 100))

    f1 = f1_score(y_test, dt_model.predict(X_test), average='weighted')
    print("Decision Tree F1 Score: %.2f" % f1)

    return dt_model, dt_score, f1

# Support Vector Machine
def svm(X_train, y_train, X_test, y_test):
    svm_model = SVC(random_state=987654321)
    print("Fitting SVM model...")
    svm_model.fit(X_train, y_train)

    svm_score = svm_model.score(X_test, y_test)
    print("SVM Accuracy: %.2f%%" % (svm_score * 100))

    f1 = f1_score(y_test, svm_model.predict(X_test), average='weighted')
    print("SVM F1 Score: %.2f" % f1)

    return svm_model, svm_score, f1

# Bagging
def bagging(X_train, y_train, X_test, y_test, dt_model):
    sss = StratifiedShuffleSplit(test_size=20, random_state=987654321)
    bagging_model = BaggingClassifier(
        estimator=dt_model, random_state=987654321)

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

    f1 = f1_score(y_test, bst_bg_model.predict(X_test), average='weighted')
    print("Bagging F1 Score: %.2f" % f1)

    return bst_bg_model, bg_score, f1

# Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=987654321, max_depth=2)
    print("Fitting Random Forest model...")
    rf_model.fit(X_train, y_train)

    rf_score = rf_model.score(X_test, y_test)
    print("Random Forest Accuracy: %.2f%%" % (rf_score * 100))

    f1 = f1_score(y_test, rf_model.predict(X_test), average='weighted')
    print("Random Forest F1 Score: %.2f" % f1)

    return rf_model, rf_score, f1

# Gradient Boosting
def gradient_boosting(X_train, y_train, X_test, y_test):
    gbc_model = GradientBoostingClassifier(random_state=987654321, max_depth=2)
    print("Fitting Gradient Boosting model...")
    gbc_model.fit(X_train, y_train)

    gbc_score = gbc_model.score(X_test, y_test)
    print("Gradient Boosting Accuracy: %.2f%%" % (gbc_score * 100))

    f1 = f1_score(y_test, gbc_model.predict(X_test), average='weighted')
    print("Gradient Boosting F1 Score: %.2f" % f1)

    return gbc_model, gbc_score, f1

# XGBoost
def xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(max_depth=2, objective='reg:squarederror')

    print("Fitting XGBoost model...")
    xgb_model.fit(X_train, y_train)

    xgb_score = xgb_model.score(X_test, y_test)
    print("XGBoost Accuracy: %.2f%%" % (xgb_score * 100))

    f1 = f1_score(y_test, xgb_model.predict(X_test), average='weighted')
    print("XGBoost F1 Score: %.2f" % f1)

    return xgb_model, xgb_score, f1

# Stacking Classifier
def stacking(X_train, y_train, X_test, y_test, model1, model2, model3):
    estimators = [("model1", model1), ("model2", model2), ("model3", model3)]
    st_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    print("Fitting Stacking model...")
    st_model.fit(X_train, y_train)

    st_score = st_model.score(X_test, y_test)
    print("Stacking Accuracy: %.2f%%" % (st_score * 100))

    f1 = f1_score(y_test, st_model.predict(X_test), average='weighted')
    print("Stacking F1 Score: %.2f" % f1)

    return st_model, st_score, f1