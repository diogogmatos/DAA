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
from models import decision_tree, svm, bagging, random_forest, gradient_boosting, xgboost, stacking
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from colorama import Fore

pd.set_option('display.max_info_columns', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_rows', 3000)


def preprocess(original_df: DataFrame, test=False) -> DataFrame:
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

    if not test:
        # drop duplicate rows
        df.drop_duplicates(inplace=True)

        # separate features and target
        X = df.drop('Transition', axis=1)
        y = df['Transition']

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        # apply standard scaling
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # apply label encoding
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        return X_train, X_test, y_train, y_test, le
    else:
        # apply standard scaling
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        return df


def feature_selection(X_train, X_test, y_train, y_test, df_test: DataFrame, N: int | float | None = None):
    rf_model, _, _ = random_forest(X_train, y_train, X_test, y_test)
    # dt_model, _, _ = decision_tree(X_train, y_train, X_test, y_test)
    # xgb_model, _, _ = xgboost(X_train, y_train, X_test, y_test)
    model = rf_model

    # select 80% of features
    rfe = RFE(estimator=model, n_features_to_select=N, step=1)
    rfe.fit(X_train, y_train)

    print(Fore.YELLOW + f"🧹 Removed {X_train.shape[1] - rfe.n_features_} features (New total: {rfe.n_features_}).")

    selected_features = X_train.columns[rfe.support_]
    X_train_rfe = X_train[selected_features]
    X_test_rfe = X_test[selected_features]
    df_test_rfe = df_test[selected_features]

    return X_train_rfe, X_test_rfe, df_test_rfe


def main():
    print(Fore.BLUE + "⏳ Loading datasets...")
    df_train = pd.read_csv('datasets/train_radiomics_hipocamp.csv')
    df_test = pd.read_csv('datasets/test_radiomics_hipocamp.csv')
    print()

    print(Fore.BLUE + "🪄 Preprocessing datasets...")
    X_train, X_test, y_train, y_test, le = preprocess(df_train)
    df_test = preprocess(df_test, test=True)
    print()

    print(Fore.BLUE + "🧐 Performing Recursive Feature Elimination..." + Fore.WHITE)
    X_train_rfe, X_test_rfe, df_test_rfe = feature_selection(
        X_train, X_test, y_train, y_test, df_test, 0.7)
    X_train = X_train_rfe
    X_test = X_test_rfe
    df_test = df_test_rfe
    print()

    print(Fore.BLUE + "⚖️ Balancing classes with SMOTE..." + Fore.WHITE)
    smote = SMOTE(random_state=987654321)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_train = X_train_smote
    y_train = y_train_smote
    print()

    print(Fore.BLUE + "🧐 Performing Principal Component Analysis" + Fore.WHITE)
    pca = PCA(n_components=0.95, random_state=987654321) # 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    df_test_pca = pca.transform(df_test)
    print(Fore.YELLOW + f"🧹 Reduced {X_train.shape[1] - pca.n_components_} features (New total: {pca.n_components_}).")
    X_train = pd.DataFrame(X_train_pca)
    X_test = pd.DataFrame(X_test_pca)
    df_test = pd.DataFrame(df_test_pca)
    print()

    print(Fore.BLUE + "⚙️ Training models..." + Fore.WHITE)
    # Run models
    dt_model, dt_score, dt_f1 = decision_tree(X_train, y_train, X_test, y_test)
    rf_model, rf_score, rf_f1 = random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_score, xgb_f1 = xgboost(X_train, y_train, X_test, y_test)
    svm_model, svm_score, svm_f1 = svm(X_train, y_train, X_test, y_test)
    # SLOW ONES
    gbc_model, gbc_score, gbc_f1 = gradient_boosting(X_train, y_train, X_test, y_test)
    bst_bg_model, bg_score, bg_f1 = bagging(X_train, y_train, X_test, y_test, dt_model)
    print()

    top_models = {
        'Decision Tree': [dt_model, dt_f1],
        'SVM': [svm_model, svm_f1],
        'Random Forest': [rf_model, rf_f1],
        'XGBoost': [xgb_model, xgb_f1],
        'Gradient Boosting': [gbc_model, gbc_f1],
        'Bagging': [bst_bg_model, bg_f1]
    }

    # sort
    top_models = {k: v for k, v in sorted(
        top_models.items(), key=lambda item: item[1][1], reverse=True)}
    top1, top2, top3 = list(top_models.items())[:3]

    print(Fore.BLUE + "🔀 Stacking top 3 models..." + Fore.WHITE)
    stack, stack_score, stack_f1 = stacking(X_train, y_train, X_test, y_test, top1[1][0], top2[1][0], top3[1][0])
    top_models['stacking'] = [stack, stack_f1]
    print()

    # sort again
    top_models = {k: v for k, v in sorted(
        top_models.items(), key=lambda item: item[1][1], reverse=True)}

    print(Fore.BLUE + "📊 Model comparison (F1):")
    i = 1
    for name, model_score in top_models.items():
        print(Fore.WHITE + f"{i}. {name.capitalize()
                                   }: {model_score[1].round(2)}")
        i += 1
    print()

    # Kaggle submissions
    print(Fore.BLUE + "📝 Generating submissions...")

    for name, model in top_models.items():

        print(Fore.WHITE + f"Generating {name} submission...")
        dt_predictions = model[0].predict(df_test)
        dt_predictions = le.inverse_transform(dt_predictions)

        submission = pd.DataFrame(
            {'RowId': df_test.index + 1, 'Result': dt_predictions})

        with open(f'results/{name.lower().replace(" ", "_")}.csv', 'w') as file:
            submission.to_csv(file, index=False, sep=",")

    print(Fore.GREEN + "\n✅ Done!")


if __name__ == "__main__":
    main()
