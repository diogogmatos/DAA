from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score, make_scorer


def train_grid_search_cv(model_name, model, param_grid, X_train, y_train, X_test, y_test, X, y):
    print(f"Training {model_name} model...")

    scoring = make_scorer(f1_score, average='weighted')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=987654321)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=cv)

    print(f"> Grid Searching...")
    grid_search.fit(X_train, y_train)

    print("- Best Parameters:")
    with open(f"data/best_params_{model_name.lower().replace(' ', '_')}.txt", "w") as file:
        for key, value in grid_search.best_params_.items():
            print(f"\t{key}: {value}")
            file.write(f"{key}:{value}\n")

    print("- Estimated Score (Avg. F1): %.2f" % grid_search.cv_results_['mean_test_score'].mean())

    print("> Evaluating best model on test data...")
    y_pred = grid_search.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"- F1 Score: {f1:.2f}")

    print("> Fitting best model on all training data...")
    best_estimator = grid_search.best_estimator_
    best_estimator.fit(X, y)
    print("Done.")

    return best_estimator, f1

def train(model_name, model, X_train, y_train, X_test, y_test, X, y):
    print(f"Training {model_name} model...")

    print("> Fitting...")
    model.fit(X_train, y_train)

    print("> Evaluating model on test data...")
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"- F1 Score: {f1:.2f}")

    print("> Fitting model on all training data...")
    model.fit(X, y)
    print("Done.")

    return model, f1

# Decision Tree
def decision_tree():
    model = DecisionTreeClassifier(random_state=987654321)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_leaf_nodes': [None, 1000],
        'max_depth': [4, 8, 10],
        'min_samples_split': [4, 6, 8],
        'min_samples_leaf': [0.5, 1, 2],
        'max_features': ['sqrt']
    }
    return model, param_grid

# Support Vector Machine
def svm():
    svm_model = SVC(random_state=987654321)
    param_grid = {
        'C': [2, 5, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    return svm_model, param_grid

# Bagging
def bagging(model):
    bagging_model = BaggingClassifier(
        estimator=model, random_state=987654321, n_jobs=-1)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }

    return bagging_model, param_grid

def customBagging():
    # TODO
    pass

# Random Forest
def random_forest():
    rf_model = RandomForestClassifier(random_state=987654321, n_jobs=-1)
    param_grid = {
        'criterion': ['gini'],
        'min_samples_split': [4, 6, 8],
        'min_samples_leaf': [1, 2, 4],
        'n_estimators': [100, 150, 200],
        'max_features': ['sqrt'],
        'max_depth': [6, 8, 10, None],
    }

    return rf_model, param_grid

# Gradient Boosting
def gradient_boosting():
    gbc_model = GradientBoostingClassifier(random_state=987654321)
    param_grid = {
        'n_estimators': [200, 250, 300],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [2, 4, 6],
    }

    return gbc_model, param_grid

# XGBoost
def xgboost():
    xgb_model = XGBClassifier(objective='reg:squarederror', random_state=987654321, n_jobs=-1)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.3, 0.5],
        'booster': ['gbtree', 'gblinear', 'dart'],
    }

    return xgb_model, param_grid

# Stacking Classifier
def stacking(model1, model2, model3):
    estimators = [("model1", model1), ("model2", model2), ("model3", model3)]
    st_model = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(max_iter=1000000), n_jobs=-1, cv=5)
    param_grid = {
        'stack_method': ['auto', 'predict'],
    }

    return st_model, param_grid
