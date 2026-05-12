import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score,
    average_precision_score, confusion_matrix
)
from scipy.stats import bootstrap
import optuna
from sklearn.base import clone
from mrmr import mrmr_classif
from preprocessing import build_preprocessor


class repeated_nested_cv:

    def __init__(self, classifiers, parameter_grids, R, N, K, seed=42, tune = False, feature_select = False):
        self.classifiers = classifiers              #List of tuples with classifiers
        self.parameter_grids = parameter_grids      #Parameter grids (dictionaries)
        self.R = R                                  #Repetitions of CV
        self.N = N                                  #Outer loop cv folds
        self.K = K                                  #Inner loop cv Folds
        self.seed = seed                            #Random seed for reproducible results
        self.results = {}                           #Stores metrics       
        self.tune = tune                            #For hyperparameter tuning or not
        self.feature_select = feature_select
        self.feature_counts = {}
        self.best_params_ = {}


    def _compute_metrics(self,y,y_pred,y_prob):
        tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()  #Get TN,FN,TP,FP values from the confusion matrix
        specificity = tn / (tn+fp)                           #Calculate specificity
        
        return {
            "MCC": matthews_corrcoef(y,y_pred),
            "AUC": roc_auc_score(y, y_prob),
            "BA": balanced_accuracy_score(y,y_pred),
            "F1": f1_score(y,y_pred),
            "Recall": recall_score(y, y_pred),
            "Specificity": specificity,
            "Precision": precision_score(y,y_pred),
            "PRAUC": average_precision_score(y,y_prob)
        }
        
    def fit(self, X, y):
        X = X.reset_index(drop = True)
        y = y.reset_index(drop = True)
        feature_names = X.columns.tolist()


        for name, classifier in self.classifiers:
            self.results[name] = []   #Create empty list for metrics of each classifier

            for r in range(self.R):
                outer_cv_folds = StratifiedKFold(n_splits=self.N, shuffle = True, random_state= self.seed + r)
            
                for train_index,test_index in outer_cv_folds.split(X, y):
                    preprocessor = build_preprocessor()
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]     

                    X_train_preprocessed = preprocessor.fit_transform(X_train)
                    X_test_preprocessed = preprocessor.transform(X_test)  #No fitting on the test fold!
                    
                    inner_cv = StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.seed)    

                    if self.feature_select:
                        best_features, best_k = self._mrmr_select(
                            X_train_preprocessed, y_train, classifier, inner_cv, feature_names
                        )

                        X_train_final = pd.DataFrame(X_train_preprocessed, columns=feature_names)[best_features]
                        X_test_final  = pd.DataFrame(X_test_preprocessed, columns=feature_names)[best_features]
                    else:
                        X_train_final = X_train_preprocessed
                        X_test_final = X_test_preprocessed     

                    if self.tune:
                        best_params = self._inner_loop(X_train_final, y_train, name, classifier, inner_cv)
                        final_clf = clone(classifier).set_params(**best_params)

                        if name not in self.best_params_:
                            self.best_params_[name] = []
                        self.best_params_[name].append(best_params)
                    
                    else:
                        final_clf = clone(classifier)

                    final_clf.fit(X_train_final, y_train)
                    y_pred = final_clf.predict(X_test_final)
                    y_prob = final_clf.predict_proba(X_test_final)[:, 1]
                    
                    metrics = self._compute_metrics(y_test, y_pred, y_prob)
                    self.results[name].append(metrics)

    def _bootstrap_ci(self, values, resamples=1000, confidence=0.95):
        data = (np.array(values),)
        result = bootstrap(data,
                            np.median,
                            n_resamples = resamples,
                            confidence_level = confidence,
                            random_state = self.seed,
                            method = "percentile")
        return result.confidence_interval.low, result.confidence_interval.high
    
    def get_results(self):
        results_summary = {}

        for name in self.results:
            results_df = pd.DataFrame(self.results[name])
            results_summary[name] = {}

            for metric in results_df.columns:
                values = results_df[metric].values
                median = np.median(values)
                low, high = self._bootstrap_ci(values)
            
                results_summary[name][metric] = {
                    "median"  : round(median, 3),
                    "CI_low"  : round(low, 3),
                    "CI_high" : round(high, 3)
                }

        return results_summary
    
    def _inner_loop(self, X_train_processed, y_train, name, classifier, inner_cv):
        X_train_processed = np.array(X_train_processed)
        
        def objective(trial):
            params = self.parameter_grids[name](trial)
            fold_scores = []

            for inner_train_index, inner_test_index in inner_cv.split(X_train_processed, y_train):
                X_inner_train = X_train_processed[inner_train_index]
                X_inner_test  = X_train_processed[inner_test_index]
                y_inner_train = y_train.iloc[inner_train_index]
                y_inner_test  = y_train.iloc[inner_test_index]

                inner_classifier = clone(classifier).set_params(**params)
                inner_classifier.fit(X_inner_train, y_inner_train)

                y_prob = inner_classifier.predict_proba(X_inner_test)[:, 1]
                score  = roc_auc_score(y_inner_test, y_prob)
                fold_scores.append(score)

            return np.mean(fold_scores)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=50)
        return study.best_params

    def _mrmr_select(self, X_train, y_train, classifier, inner_cv, feature_names, K_values = range(3,14)):
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        y_train = y_train.reset_index(drop=True)
        best_auc = -np.inf
        best_k = None
        best_features = None

        for k in K_values:
            selected = mrmr_classif(X=X_train_df, y=y_train, K=k)

            fold_scores = []
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_inner_train = X_train_df.iloc[inner_train_idx][selected]
                X_inner_val   = X_train_df.iloc[inner_val_idx][selected]
                y_inner_train = y_train.iloc[inner_train_idx]
                y_inner_val   = y_train.iloc[inner_val_idx]

                clf = clone(classifier)
                clf.fit(X_inner_train, y_inner_train)
                y_prob = clf.predict_proba(X_inner_val)[:, 1]
                fold_scores.append(roc_auc_score(y_inner_val, y_prob))

            mean_auc = np.mean(fold_scores)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_k = k
                best_features = selected

        # track feature selection frequency
        for f in best_features:
            self.feature_counts[f] = self.feature_counts.get(f, 0) + 1

        return best_features, best_k
    
    def get_feature_frequencies(self, feature_names):
        total_folds = self.R * self.N  # 10 * 5 = 50
        freq_df = pd.DataFrame({
            'feature':   list(self.feature_counts.keys()),
            'count':     list(self.feature_counts.values()),
            'frequency': [v / total_folds for v in self.feature_counts.values()]
        })
        return freq_df.sort_values('frequency', ascending=False).reset_index(drop=True)
    

    def get_best_params(self):
        summary = {}
        for name, params_list in self.best_params_.items():
            params_df = pd.DataFrame(params_list)
            summary[name] = {}

            for param in params_df.columns:
                #First for numerical params across folds
                if params_df[param].dtype in [np.float64, np.int64]:
                    summary[name][param] = {
                        'median': round(params_df[param].median(), 4),
                        'min':    round(params_df[param].min(), 4),
                        'max':    round(params_df[param].max(), 4)
                    }
                else:
                    # for categorical params show most common value
                    summary[name][param] = params_df[param].mode()[0]
        return summary