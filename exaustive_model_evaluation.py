import json
import sys
import pandas as pd
from generate_workload import models
from preprocessing import preprocess, feature_selection_balancing
from models import train_grid_search_cv

if __name__=="__main__":
    if len(sys.argv) < 2:
        version = input("please sepcify a version: ")
    else:
        version = sys.argv[1]
    
    workload = dict()
    with open(f"exaustive_model_evaluation/{version}/workload.json","r") as f:
        workload = json.load(f)
    
    df_train = pd.read_csv('datasets/train_radiomics_hipocamp.csv')
    df_test = pd.read_csv('datasets/test_radiomics_hipocamp.csv')

    X_train, X_test, y_train, y_test, le = preprocess(df_train)
    X, y = preprocess(df_train, mode="all")
    df_test = preprocess(df_test, mode="test")

    for i,model_info in enumerate(workload):
        if model_info.get('done'):
            continue
        else:
            model_name = model_info['model']
            model_func = models[model_info['model']]['model']
            model,param_grid = model_func()

            X_train, X_test, y_train, y_test, X, df_test = feature_selection_balancing(
                X_train, X_test, y_train, y_test, X, y, df_test,arg_settings=set(model_info['treatment']))

            model, score,report,params = train_grid_search_cv(
                model_name, model, param_grid, X_train, y_train, X_test, y_test, X, y)
            
            replace = []
            ks = [int(k) for k in report.keys() if k.isdigit()]
            k_primes = le.inverse_transform(ks)
            print("REPORT",report)
            print(list(zip(ks,k_primes)))
            for k,k_prime in zip(ks,k_primes):
                report[k_prime] = report[str(k)]
                del report[str(k)]

            workload[i]['done'] = True
            workload[i]['results'] = report
            workload[i]['params'] = params

            with open(f"exaustive_model_evaluation/{version}/workload.json","w") as f:
                workload = json.dump(workload,f,indent=2)
            