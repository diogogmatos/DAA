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
from models import decision_tree, svm, bagging, random_forest, gradient_boosting, xgboost, stacking, train_grid_search_cv, train
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from colorama import Fore
from simple_term_menu import TerminalMenu

pd.set_option('display.max_info_columns', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)

SETTINGS = []

def preprocess(original_df: DataFrame, mode="normal") -> DataFrame:
    df = original_df.copy()

    # drop columns that always have the same value
    unique_dict = df.nunique().to_dict()
    no_unique_values = {col: count for col,
                        count in unique_dict.items() if count == 1}
    drop = [col for col, _ in no_unique_values.items()]
    df.drop(drop, axis=1, inplace=True)

    # drop categorical columns that always have unique values
    all_unique = [col for col, count in unique_dict.items(
    ) if count == df.shape[0] and df[col].dtype == 'object']
    df.drop(all_unique, axis=1, inplace=True)

    scaler = StandardScaler()
    le = LabelEncoder()

    if mode == "normal":
        # drop duplicate rows
        df.drop_duplicates(inplace=True)

        # separate features and target
        X = df.drop('Transition', axis=1)
        y = df['Transition']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=987654321)

        # apply standard scaling
        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # apply label encoding
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        return X_train, X_test, y_train, y_test, le
    elif mode == "test":
        # apply standard scaling
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        return df
    elif mode == "all":
        # separate features and target
        X = df.drop('Transition', axis=1)
        y = df['Transition']

        # apply standard scaling
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # apply label encoding
        y = le.fit_transform(y)

        return X, y


def rfe(X_train, X_test, y_train, y_test, X, y, df_test: DataFrame, N: int | float | None = None):
    global SETTINGS
    
    selected_features = None

    # check for cached selected features
    if "Use cached RFECV selection data." in SETTINGS:
        try:
            with open('data/rfecv_selected_features.txt', 'r') as file:
                selected_features = file.read().splitlines()
                print("- 📁 Found cached selection.")
                print(Fore.YELLOW + f"- 🧹 Removed {X_train.shape[1] - len(
                    selected_features)} features (New total: {len(selected_features)}).")
        except FileNotFoundError:
            pass

    if not selected_features:
        print()
        rf_model = RandomForestClassifier(
            max_depth=6, max_features='sqrt', n_estimators=200, random_state=987654321, n_jobs=-1)
        model, _ = train('Random Forest', rf_model, X_train,
                         y_train, X_test, y_test, X, y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=987654321)
        rfe = RFECV(estimator=model, step=1, cv=cv,
                    scoring='f1_macro', n_jobs=-1)
        rfe.fit(X_train, y_train)

        print(Fore.YELLOW + f"- 🧹 Removed {
              X_train.shape[1] - rfe.n_features_} features (New total: {rfe.n_features_}).")

        selected_features = X_train.columns[rfe.support_]

        # cache selected features
        with open('data/rfecv_selected_features.txt', 'w') as file:
            for feature in selected_features:
                file.write(f"{feature}\n")

    X_train_rfe = X_train[selected_features]
    X_test_rfe = X_test[selected_features]
    df_test_rfe = df_test[selected_features]
    X_rfe = X[selected_features]

    return X_train_rfe, X_test_rfe, X_rfe, df_test_rfe


def feature_engineering(X_train, X_test, y_train, y_test, X, y, df_test):
    if "RFECV" in SETTINGS:
        print(Fore.BLUE + "> 🧐 Performing Recursive Feature Elimination..." + Fore.WHITE)
        X_train_rfe, X_test_rfe, X_rfe, df_test_rfe = rfe(
            X_train, X_test, y_train, y_test, X, y, df_test, 0.7)
        X_train = X_train_rfe
        X_test = X_test_rfe
        X = X_rfe
        df_test = df_test_rfe
        print()

    if "SMOTE" in SETTINGS:
        print(Fore.BLUE + "> ⚖️ Balancing classes with SMOTE..." + Fore.WHITE)
        smote = SMOTE(random_state=987654321)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        X_train = X_train_smote
        y_train = y_train_smote
        print()

    if "PCA" in SETTINGS:
        print(Fore.BLUE + "> 🧐 Performing Principal Component Analysis" + Fore.WHITE)
        pca = PCA(n_components=0.995, random_state=987654321) # 99% of variance
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        df_test_pca = pca.transform(df_test)
        X_pca = pca.transform(X)
        print(Fore.YELLOW + f"- 🧹 Reduced {X_train.shape[1] - pca.n_components_} features (New total: {pca.n_components_}).")
        X_train = pd.DataFrame(X_train_pca)
        X_test = pd.DataFrame(X_test_pca)
        df_test = pd.DataFrame(df_test_pca)
        X = pd.DataFrame(X_pca)
        print()

    return X_train, X_test, y_train, y_test, X, df_test


def model_comparisson(model_results):
    print(Fore.BLUE + "📊 Model comparison (F1):")
    i = 1
    for name, model_score in model_results.items():
        print(Fore.WHITE + f"{i}. {name.capitalize()
                                   }: {model_score[1].round(2)}")
        i += 1
    print()


def execute(selected_option, selected_models, single=False):
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
    X_train, X_test, y_train, y_test, X, df_test = feature_engineering(
        X_train, X_test, y_train, y_test, X, y, df_test)

    print(Fore.BLUE + "⚙️ Training...\n" + Fore.WHITE)
    # run models
    model_results = {}
    for model_name, model_function in selected_models.items():
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
    if not single:
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
                        execute(selected_option, selected_models, single=True)
                    else:   
                        back2 = True
            else:
                selected_models = groups[selected_option]
                execute(selected_option, selected_models)
        else:
            back1 = True


def read_settings():
    try:
        with open("data/settings.txt", "r") as file:
            return file.read().splitlines()
    except FileNotFoundError:
        with open("data/settings.txt", "x") as file:
            file.write()
            return []
        

def write_settings(selected):
    global SETTINGS
    with open("data/settings.txt", "w") as file:
        for selection in selected:
            file.write(f"{selection}\n")
    SETTINGS = selected


def settings():
    global SETTINGS
    SETTINGS = read_settings()
    options = ["RFECV", "SMOTE", "PCA", "Use cached RFECV selection data."]
    menu = TerminalMenu(options, multi_select=True, multi_select_empty_ok=True, preselected_entries=SETTINGS, show_multi_select_hint_text="Press SPACE to select, ENTER to confirm.", show_multi_select_hint=True, title="Settings", multi_select_select_on_accept=False)
    menu.show()
    selected = []
    if menu.chosen_menu_entries:
        selected = [x for x in menu.chosen_menu_entries]
    write_settings(selected)


def main():
    global SETTINGS
    SETTINGS = read_settings()

    exit = False
    while not exit:
        options = ["Run Models", "Settings", "Exit"]
        menu = TerminalMenu(options, title="Welcome")
        index = menu.show()
        if index == 0:
            run_models()
        elif index == 1:
            settings()
        elif index == 2 or index == None:
            exit = True


if __name__ == "__main__":
    main()
