import pandas as pd
import os
from ..models import *
from itertools import chain, combinations
import json

command = 'kaggle competitions submit -c sbsppdaa24 -f submission.csv -m "Message"'

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return set(chain.from_iterable(set(combinations(s, r)) for r in range(len(s) + 1)))


def mkdir_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

VERSION = "0.0"

if __name__ == "__main__":
    mkdir_if_not_exists("exaustive_model_evaluation")
    models = {
        "Decision Tree": {
            "model": decision_tree,
            "fast": True,
            "incompatible": {"PCA"},
        },
        "Random Forest": {
            "model": random_forest,
            "fast": True,
            "incompatible": {"PCA"},
        },
        "XGBoost": {"model": xgboost, "incompatible": set()},
        "SVM": {"model": svm, "fast": True, "incompatible": set()},
        "Gradient Boosting": {"model": gradient_boosting, "incompatible": set()},
        "Bagging": {"model": bagging, "incompatible": set()},
    }

    treatments = {"RFECV", "SMOTE", "PCA"}

    treatments_incompatibility = {"PCA":{"RFECV"},"SMOTE":set(),'RFECV':set()}
    to_update = dict()
    for k,values in treatments_incompatibility.items():
        for v in values:
            a = treatments_incompatibility.get(v)
            if a == None or a == set():
                a = {k}
            else:
                a |= {k}
            to_update.update({v:a})
    treatments_incompatibility.update(to_update)

    treatments_ordering = {"RFECV": "SMOTE", "PCA": "SMOTE"}
    """A:B => A antes de B"""

    df_train = pd.read_csv("datasets/train_radiomics_hipocamp.csv")
    df_test = pd.read_csv("datasets/test_radiomics_hipocamp.csv")
    l = []
    for model_name, model_dict in models.items():
        for combination in powerset(treatments - model_dict["incompatible"]):
            treatment = set(combination)
            checking = [(x,treatment - set([x])) for x in treatment if set(x) != treatment]
            can = True
            for pair in checking:
                if treatments_incompatibility[pair[0]] & pair[1]:
                    can = False
            if not can:
                continue

            l.append({'model':model_name,'treatment':list(treatment)})
    
    print(json.dumps(l,indent=2))


    