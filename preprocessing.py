import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from models import train
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from colorama import Fore


import settings


def drop_constant(df):
    "drop columns that always have the same value"
    unique_dict = df.nunique().to_dict()
    no_unique_values = {col: count for col,
                        count in unique_dict.items() if count == 1}
    drop = [col for col, _ in no_unique_values.items()]
    df.drop(drop, axis=1, inplace=True)
    return df

def drop_unique(df):
    "drop categorical columns that always have unique values"
    
    unique_dict = df.nunique().to_dict()
    all_unique = [col for col, count in unique_dict.items(
    ) if count == df.shape[0] and df[col].dtype == 'object']
    df.drop(all_unique, axis=1, inplace=True)
    return df


def drop_duplicate(df):
    # drop identical columns
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups = []
    for t, v in groups.items():
        dcols = df[v].to_dict(orient="list")

        vs = list(dcols.values())
        ks = list(dcols.keys())
        lvs = len(vs)

        for i in range(lvs):
            for j in range(i+1, lvs):
                if vs[i] == vs[j]:
                    dups.append(ks[i])
                    break

    df.drop(dups, axis=1, inplace=True)
    return df


def preprocess(original_df: DataFrame, mode="normal"):
    df = original_df.copy()
    # drop columns that always have the same value
    df = drop_constant(df)

    # drop categorical columns that always have unique values
    df = drop_unique(df)

    # drop identical columns
    df = drop_duplicate(df)

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
            X, y, test_size=0.2, random_state=987654321, stratify=y)

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
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        return df
    elif mode == "all":
        # drop duplicate rows
        df.drop_duplicates(inplace=True)
        
        # separate features and target
        X = df.drop('Transition', axis=1)
        y = df['Transition']

        # apply standard scaling
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # apply label encoding
        y = pd.DataFrame(le.fit_transform(y))

        return X, y


def rfe(X_train, X_test, y_train, y_test, X, y, df_test: DataFrame, N: int | float | None = None):
    selected_features = None

    # check for cached selected features
    if "Use cached RFECV selection data." in settings.SELECTED:
        try:
            with open('data/rfecv_selected_features.txt', 'r') as file:
                selected_features = file.read().splitlines()
                print("- 📁 Found cached selection.")
                print(Fore.YELLOW + f"- 🧹 Removed {X_train.shape[1] - len(selected_features)} features (New total: {len(selected_features)}).")
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

        print(Fore.YELLOW + f"- 🧹 Removed {X_train.shape[1] - rfe.n_features_} features (New total: {rfe.n_features_}).")

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


def feature_selection_balancing(X_train, X_test, y_train, y_test, X, y, df_test,arg_settings = {}):
    if   "RFECV" in arg_settings if arg_settings else "RFECV" in settings.SELECTED  :
        print(Fore.BLUE + "> 🧐 Performing Recursive Feature Elimination..." + Fore.WHITE)
        X_train_rfe, X_test_rfe, X_rfe, df_test_rfe = rfe(
            X_train, X_test, y_train, y_test, X, y, df_test, 0.7)
        X_train = X_train_rfe
        X_test = X_test_rfe
        X = X_rfe
        df_test = df_test_rfe
        print()

    if "SMOTE" in arg_settings if arg_settings else "SMOTE" in settings.SELECTED:
        print(Fore.BLUE + "> ⚖️ Balancing classes with SMOTE..." + Fore.WHITE)
        smote = SMOTE(random_state=987654321)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        X_train = X_train_smote
        y_train = y_train_smote
        print()

    # since this is a non-linear problem, PCA is not recommended
    if "PCA" in arg_settings if arg_settings else "PCA" in settings.SELECTED:
        print(Fore.BLUE + "> 🧐 Performing Principal Component Analysis" + Fore.WHITE)
        # 99% of variance
        pca = PCA(n_components=0.995, random_state=987654321)
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
