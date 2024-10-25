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
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from pandas import DataFrame
from models import decision_tree, svm, bagging, random_forest, gradient_boosting, xgboost, stacking
from sklearn.preprocessing import LabelEncoder

def preprocess(original_df: DataFrame, test=False) -> DataFrame:
    df = original_df.copy()

    # drop duplicate rows
    if not test:
        df.drop_duplicates(inplace=True)

    # drop columns that always have the same value
    unique_dict = df.nunique().to_dict()
    no_unique_values = {col: count for col,
                        count in unique_dict.items() if count == 1}
    drop = [col for col, _ in no_unique_values.items()]
    df.drop(drop, axis=1, inplace=True)

    # drop columns that always have unique values, except the useful ones
    all_unique = [col for col, count in unique_dict.items(
    ) if count == df.shape[0] and df[col].dtype == 'object']
    df.drop(list(set(all_unique) - set(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'])), axis=1, inplace=True)

    # calculate volumes from bounding box coordinates
    volumes = []
    i = 0

    for value in df['diagnostics_Mask-original_BoundingBox']:
        coordinates = value.split(', ')
        x_max, y_max, z_max, x_min, y_min, z_min = [
            int(coordinate.replace("(", "").replace(")", "")) for coordinate in coordinates]
        # drop rows with invalid bounding box coordinates (max <= min)
        if (x_max <= x_min or y_max <= y_min or z_max <= z_min) and not test:
            df.drop(i, axis=0, inplace=True)
        else:
            volumes.append((x_max - x_min) * (y_max - y_min) * (z_max - z_min))
        i += 1

    # replace coordinates with volumes in new column
    df.drop(['diagnostics_Mask-original_BoundingBox'], axis=1, inplace=True)
    df['BoundingBox_Volume'] = volumes

    # split CenterOfMassIndex coordinates into separate columns
    com_x = []
    com_y = []
    com_z = []

    for value in df['diagnostics_Mask-original_CenterOfMass']:
        coordinates = value.split(', ')
        x, y, z = [float(coordinate.replace("(", "").replace(")", ""))
                   for coordinate in coordinates]
        com_x.append(x)
        com_y.append(y)
        com_z.append(z)

    df.drop(['diagnostics_Mask-original_CenterOfMass'],
            axis=1, inplace=True)
    df['CenterOfMass_X'] = com_x
    df['CenterOfMass_Y'] = com_y
    df['CenterOfMass_Z'] = com_z

    return df

print("Loading datasets...")
df_train = pd.read_csv('datasets/train_radiomics_hipocamp.csv')
df_test = pd.read_csv('datasets/test_radiomics_hipocamp.csv')

print("Preprocessing datasets...")
df_train = preprocess(df_train)
df_test = preprocess(df_test, test=True)

print("Training models...")
X = df_train.drop('Transition', axis=1)
y = df_train['Transition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2024)

# Run models
dt_model = decision_tree(X_train, y_train, X_test, y_test)
svm_model = svm(X_train, y_train, X_test, y_test)
#bst_bg_model = bagging(X_train, y_train, X_test, y_test, dt_model)
rf_model = random_forest(X_train, y_train, X_test, y_test)
gbc_model = gradient_boosting(X_train, y_train, X_test, y_test)
xgb_model, le = xgboost(X_train, y_train, X_test, y_test)
stacking = stacking(X_train, y_train, X_test, y_test, dt_model, rf_model, gbc_model)

# Kaggle submissions
print("Generating submissions...")

for name, model in zip(['decision_tree', 'svm', 'random_forest', 'gradient_boosting', 'xgboost', 'stacking'], [dt_model, svm_model, rf_model, gbc_model, xgb_model, stacking]):
    print(f"Generating {name} submission...")

    dt_predictions = model.predict(df_test)
    if name == 'xgboost':
        dt_predictions = le.inverse_transform(dt_predictions)
    submission = pd.DataFrame(
        {'RowId': df_test.index + 1, 'Result': dt_predictions})

    with open(f'results/{name}.csv', 'w') as file:
        submission.to_csv(file, index=False, sep=",")
