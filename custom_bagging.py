
from typing import Any, Callable, ClassVar, Literal, Mapping, TypeVar
import pandas as pd

import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score,f1_score, classification_report
from sklearn.utils.validation import check_is_fitted

Model = TypeVar("Model")
"modelo"


class CustomBAgging(sklearn.base.BaseEstimator):
    def __init__(
            self,
            cols:list[str],
            estimator_cols_pairs:list[tuple[sklearn.base.BaseEstimator | Model, list[str]]],
            meta_learner : sklearn.base.BaseEstimator = None,
            random_state:int=2025,
            weak_estimator_test_ratio = .2,
            weight_func :Callable = None):
        self.models = dict()
        self.estimator_cols_pairs = estimator_cols_pairs
        "`[(model1,['col1','col2',...]),...]`"
        self.cols = cols
        "column names"
        self._sub_learners_fitted = False
        self._is_fitted = False
        # TODO: verificar coerencia de colunas      
        self.random_state = random_state
        self.weak_estimator_test_ratio = weak_estimator_test_ratio
        self.weight_func = weight_func
        self.meta_learner = meta_learner
        
    def __sklearn_is_fitted__(self):
        return self._is_fitted
    
    def _df_list(self,df:pd.DataFrame)-> list[tuple[Model, pd.DataFrame]]:
        return [(estimator,df[g]) for (estimator,g) in self.estimator_cols_pairs]

    def modelPredictions(self, X:pd.DataFrame):
        if not self._sub_learners_fitted:
            raise Exception("The model hasent been fitted")
        predictions = {}
        for i, (estimator,cols) in enumerate(self.estimator_cols_pairs):
            y_pred = estimator.predict(X[cols])
            predictions[f'prediction_{i}'] = y_pred 
        return pd.DataFrame(predictions)


    def fit(self,X:pd.DataFrame,y:Any = None):
        for i, (estimator,cols) in enumerate(self.estimator_cols_pairs):
            estimator.fit(X[cols], y)
            self.estimator_cols_pairs[i] = (estimator,cols)
        self._sub_learners_fitted = True
        model_predictions = self.modelPredictions(X)
        self.meta_learner.fit(model_predictions,y)
        self._is_fitted = True
        return self


    def predict(self,X:pd.DataFrame):
        if not self._is_fitted:
            raise Exception("The model hasent been fitted")
        model_predictions = self.modelPredictions(X)
        return self.meta_learner.predict(model_predictions)

class RadiomicsBAgging(CustomBAgging):
    pass