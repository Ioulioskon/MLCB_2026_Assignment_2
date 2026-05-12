import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def display_results(results, title):
    rows = []
    for clf_name, metrics in results.items():
        row = {'Classifier': clf_name}
        for metric, vals in metrics.items():
            row[metric] = f"{vals['median']:.3f} ({vals['CI_low']:.3f}–{vals['CI_high']:.3f})"
        rows.append(row)
    
    df = pd.DataFrame(rows).set_index('Classifier')
    print(f"\n{title}\n{'='*60}")
    return df


def plot_metric_boxplots(ncv_object, metrics_to_plot, title):
    rows = []
    for clf_name in ncv_object.results:
        for fold_metrics in ncv_object.results[clf_name]:
            row = {'Classifier': clf_name}
            row.update(fold_metrics)
            rows.append(row)
    df_long = pd.DataFrame(rows)

    n_cols = 2
    n_rows = (len(metrics_to_plot) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten()

    classifiers = df_long['Classifier'].unique()

    for i, metric in enumerate(metrics_to_plot):
        sns.boxplot(
            data=df_long,
            x='Classifier',
            y=metric,
            ax=axes[i],
            palette='Blues'
        )

        for j, clf_name in enumerate(classifiers):
            values = df_long[df_long['Classifier'] == clf_name][metric].values
            median = np.median(values)
            low, high = ncv_object._bootstrap_ci(values)

            axes[i].errorbar(
                x=j,
                y=median,
                yerr=[[median - low], [high - median]],
                fmt='none',
                color='red',
                capsize=5,
                linewidth=2,
                label='95% CI' if j == 0 else ''
            )

        axes[i].set_title(metric, fontsize=13, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].spines[['top', 'right']].set_visible(False)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        axes[i].axhline(0.5, color='grey', linestyle='--',
                        linewidth=1, alpha=0.5)
        axes[i].legend(fontsize=8)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'../figures/{title.replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def display_best_params(best_params):
    rows = []
    for clf_name, params in best_params.items():
        for param, vals in params.items():
            if isinstance(vals, dict):
                rows.append({
                    'Classifier': clf_name,
                    'Parameter':  param,
                    'Median':     vals['median'],
                    'Min':        vals['min'],
                    'Max':        vals['max']
                })
            else:
                rows.append({
                    'Classifier': clf_name,
                    'Parameter':  param,
                    'Median':     vals,
                    'Min':        '-',
                    'Max':        '-'
                })

    df = pd.DataFrame(rows).set_index(['Classifier', 'Parameter'])
    return df

def compare_full_vs_fs(full_results, fs_results, classifier_name, metrics=None):
    if metrics is None:
        metrics = ['MCC', 'AUC', 'Recall', 'Specificity', 'BA', 'F1']
    
    rows = []
    for metric in metrics:
        full = full_results[classifier_name][metric]
        fs   = fs_results[classifier_name][metric]
        diff = round(fs['median'] - full['median'], 3)
        rows.append({
            'Metric':   metric,
            'Full':     f"{full['median']:.3f} ({full['CI_low']:.3f}–{full['CI_high']:.3f})",
            'Selected': f"{fs['median']:.3f} ({fs['CI_low']:.3f}–{fs['CI_high']:.3f})",
            'Δ':        f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
        })
    
    df = pd.DataFrame(rows).set_index('Metric')
    return df

def tune_final_model(X,y, classifier, param_grid, seed=42, n_trials=50):
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

    def objective(trial):
        params = param_grid(trial)
        scores = []

        for train_index, test_index in cv.split(X,y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            preprocessor = build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            clf = clone(classifier).set_params(**params)
            clf.fit(X_train_proc, y_train)
            y_prob = clf.predict_proba(X_test_proc)[:,1]
            scores.append(roc_auc_score(y_test,y_prob))

        return np.mean(scores)
                          

    study = optuna.create_study(
        direction = "maximize",
        sampler = optuna.samplers.TPESampler(seed = seed)
    )
    study.optimize(objective, n_trials = n_trials)

    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    return study.best_params

def plot_feature_frequencies(freq_df, title='Feature Selection Frequency'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2C6FAC' if f >= 0.7 else '#A8C8E8' 
              for f in freq_df['frequency']]
    
    bars = ax.barh(freq_df['feature'], freq_df['frequency'], 
                   color=colors)
    
    for bar, val in zip(bars, freq_df['frequency']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9)
    
    ax.axvline(0.7, color='red', linestyle='--', 
               linewidth=1.5, label='70% threshold')
    ax.set_xlabel('Selection Frequency', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'../figures/{title.replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()