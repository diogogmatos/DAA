import sklearn as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from colorama import Fore
from simple_term_menu import TerminalMenu


import settings
from models import decision_tree, svm, bagging, random_forest, gradient_boosting, xgboost, stacking, train_grid_search_cv, train
from preprocessing import preprocess, feature_selection_balancing
import ann as neural_network


def model_comparisson(model_results):
    print(Fore.BLUE + "📊 Model comparison (F1):")
    i = 1
    for name, model_score in model_results.items():
        print(Fore.WHITE + f"{i}. {name.capitalize()
                                   }: {model_score[1]:.2f}")
        i += 1
    print()


def execute(selected_option, selected_models, ann=False):
    print(Fore.MAGENTA + f"🚀 Running {selected_option} model{
        "s" if len(selected_models) > 1 else ""}...\n")

    print(Fore.BLUE + "⏳ Loading datasets...")
    df_train = pd.read_csv('datasets/train_radiomics_hipocamp.csv')
    df_test = pd.read_csv('datasets/test_radiomics_hipocamp.csv')
    print()

    print(Fore.BLUE + "🪄 Preprocessing datasets...")
    X_train, X_test, y_train, y_test, le = preprocess(df_train)
    X, y = preprocess(df_train, mode="all")
    df_test = preprocess(df_test, mode="test")
    print()

    print(Fore.BLUE + "🔍 Feature engineering...\n")
    X_train, X_test, y_train, y_test, X, df_test = feature_selection_balancing(
        X_train, X_test, y_train, y_test, X, y, df_test)

    print(Fore.BLUE + "⚙️ Training...\n" + Fore.WHITE)
    # run models
    model_results = {}
    for model_name, model_function in selected_models.items():
        if ann:
            model = model_function(X.shape[1], len(le.classes_))
            train_dl, test_dl = neural_network.prepare_train_data(X, y, 0.33)
            model, score = neural_network.train_model(train_dl, test_dl, model, epochs=50, lr=0.001)
            model_results[model_name] = [model, score]
        else:
            model, param_grid = model_function()
            model, score = train_grid_search_cv(
                model_name, model, param_grid, X_train, y_train, X_test, y_test, X, y)
            model_results[model_name] = [model, score]
        print()
    print()

    # sort
    model_results = {k: v for k, v in sorted(
        model_results.items(), key=lambda item: item[1][1], reverse=True)}

    # print results
    model_comparisson(model_results)

    # stack top 3 models
    if len(selected_models) >= 3:
        model_options = ["Yes", "No"]
        menu = TerminalMenu(
            model_options, title="Stack top 3 best performing models?")
        index = menu.show()
        stack = model_options[index] == "Yes"

        if stack:
            print(Fore.BLUE + "🔀 Stacking top 3 models..." + Fore.WHITE)
            top1, top2, top3 = list(model_results.items())[:3]
            stack, param_grid = stacking(
                top1[1][0], top2[1][0], top3[1][0])
            stack, stack_score = train_grid_search_cv(
                'Stacking', stack, param_grid, X_train, y_train, X_test, y_test, X, y)
            model_results['stacking'] = [stack, stack_score]
            print()

            # sort again
            model_results = {k: v for k, v in sorted(
                model_results.items(), key=lambda item: item[1][1], reverse=True)}

            # print results
            model_comparisson(model_results)

    # Kaggle submissions
    print(Fore.BLUE + "📝 Generating submissions...")

    for name, model in model_results.items():

        print(Fore.WHITE + f"Generating {name} submission...")
        dt_predictions = None
        if ann:
            dt_predictions = neural_network.get_predictions(model[0], df_test)
        else:
            dt_predictions = model[0].predict(df_test)
        dt_predictions = le.inverse_transform(dt_predictions)

        submission = pd.DataFrame(
            {'RowId': df_test.index + 1, 'Result': dt_predictions})

        with open(f'results/{name.lower().replace(" ", "_")}.csv', 'w') as file:
            submission.to_csv(file, index=False, sep=",")

    print(Fore.GREEN + "\n✅ Done!\n" + Fore.WHITE)


def run_models():
    models = {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "SVM": svm,
        "Gradient Boosting": gradient_boosting,
        "Bagging": bagging
    }
    groups = {
        "All": {
            "Decision Tree": decision_tree,
            "Random Forest": random_forest,
            "XGBoost": xgboost,
            "SVM": svm,
            "Gradient Boosting": gradient_boosting
        },
        "Fast": {
            "Decision Tree": decision_tree,
            "Random Forest": random_forest,
            "XGBoost": xgboost,
            "SVM": svm
        },
        "Slow": {
            "Gradient Boosting": gradient_boosting
        }
    }

    back1 = False
    while not back1:
        selected_models = {}
        group_options = ["<-"] + [k for k in groups.keys()] + ["Individual"]
        menu = TerminalMenu(
            group_options, title="Models to run:")
        index = menu.show()
        if index == None:
            break
        selected_option = group_options[index]
        single = selected_option == "Individual"
        back1 = selected_option == "<-"

        if not back1:
            if single:               
                back2 = False
                while not back2:
                    model_options = [k for k in models.keys()]
                    menu = TerminalMenu(
                        ["<-"] + model_options, title="Select a model to run:", multi_select=True, show_multi_select_hint_text="Press SPACE to select, ENTER to confirm.", show_multi_select_hint=True, multi_select_select_on_accept=False)
                    indexes = menu.show()

                    if not 0 in indexes or indexes == None:
                        for index in indexes:
                            selected_models[model_options[index-1]] = models[model_options[index-1]]
                        selected_option = ", ".join(selected_models.keys())
                        execute(selected_option, selected_models)
                    else:   
                        back2 = True
            else:
                selected_models = groups[selected_option]
                execute(selected_option, selected_models)
        else:
            back1 = True


def run_ann_models():
    models = {
        "MLP": neural_network.mlp
    }
    groups = {
        "All": {
            "MLP": neural_network.mlp
        }
    }

    back1 = False
    while not back1:
        selected_models = {}
        group_options = ["<-"] + [k for k in groups.keys()] + ["Individual"]
        menu = TerminalMenu(
            group_options, title="Models to run:")
        index = menu.show()
        if index == None:
            break
        selected_option = group_options[index]
        single = selected_option == "Individual"
        back1 = selected_option == "<-"

        if not back1:
            if single:
                back2 = False
                while not back2:
                    model_options = [k for k in models.keys()]
                    menu = TerminalMenu(
                        ["<-"] + model_options, title="Select a model to run:", multi_select=True, show_multi_select_hint_text="Press SPACE to select, ENTER to confirm.", show_multi_select_hint=True, multi_select_select_on_accept=False)
                    indexes = menu.show()

                    if not 0 in indexes or indexes == None:
                        for index in indexes:
                            selected_models[model_options[index-1]] = models[model_options[index-1]]
                        selected_option = ", ".join(selected_models.keys())
                        execute(selected_option, selected_models, ann=True)
                    else:
                        back2 = True
            else:
                selected_models = groups[selected_option]
                execute(selected_option, selected_models, ann=True)



def main():
    exit = False
    while not exit:
        options = ["Run Models", "Run ANN Models", "Settings", "Exit"]
        menu = TerminalMenu(options, title="Welcome")
        index = menu.show()
        if index == 0:
            run_models()
        elif index == 1:
            run_ann_models()
        elif index == 2:
            settings.show()
        elif index == 3 or index == None:
            exit = True


if __name__ == "__main__":
    main()
