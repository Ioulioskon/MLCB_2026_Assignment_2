#Functions file for EDA for the second MLCB assignment

#IMPORTS
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

def overview(dataframe):

    print("Peek into the dataset")
    display(dataframe.head())
    

    #Summary
    n_rows, n_cols = dataframe.shape
    n_duplicates = dataframe.duplicated().sum()
    n_missing = dataframe.isnull().sum().sum()

    print("1. General Summary:")

    summary_df = pd.DataFrame({
        'Metric': ['Rows', 'Columns', 'Duplicate Rows', 'Total Missing Values'],
        'Value':  [n_rows, n_cols, n_duplicates, n_missing]
    })
    display(summary_df)


    #Data types
    print("2. Data Types:")
    n_numerical = dataframe.select_dtypes(include='number').shape[1]
    n_categorical = dataframe.select_dtypes(exclude='number').shape[1]

    print(f"Numerical features: {n_numerical}")
    print(f"Categorical features: {n_categorical}")
    display(pd.DataFrame(dataframe.dtypes, columns = ["dtype"]))

    #Descriptive statistics
    print("3.Descriptive Statistics of numerical features")
    display(dataframe.describe().T)

    #Missing values
    missing = dataframe.isna().sum()
    missing_pct = (missing / len(dataframe) * 100).round(2)

    print("4. Missing Values per feature")
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        missing_df = pd.DataFrame({
            'missing_count': missing[missing > 0],
            'missing_%':     missing_pct[missing > 0]
        })
        display(missing_df)


def find_outliers(dataframe):
    """
    Detects outliers in numerical continuous columns using the IQR (Interquantile range) method
    """
    
    #Define how many unique values are neede to detect a value as continuous
    unique_threshold = 10

    #Auto-detect numerical columns
    numerical_cols = dataframe.select_dtypes(include="number").columns.tolist()

    #Detect continuous columns
    continuous_cols = []

    for col in numerical_cols:
        if dataframe[col].nunique() > unique_threshold:
            continuous_cols.append(col)

    print(f"Continuous columns detected: {continuous_cols}")
    
    #Find outliers
    results = []

    for col in continuous_cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        n_outliers = dataframe[(dataframe[col] < lower_bound) | 
                                (dataframe[col] > upper_bound)].shape[0]
        
        outlier_values = dataframe[
            (dataframe[col] < lower_bound) | 
            (dataframe[col] > upper_bound)
        ][col].values.tolist()

        results.append({
            'feature':     col,
            'Q1':          round(Q1, 2),
            'Q3':          round(Q3, 2),
            'IQR':         round(IQR, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'n_outliers':  n_outliers,
            'outlier_%':   round(n_outliers / len(dataframe) * 100, 2),
            'outlier_values' : outlier_values
        })
    
    results_df = pd.DataFrame(results).set_index('feature')
    display(results_df)    

def class_imbalance_binary(dataframe, target = "num"):
    counts = dataframe[target].value_counts()
    pct = (counts/len(dataframe) * 100).round(2)

    imbalance_df = pd.DataFrame({
        "Class": counts.index,
        "Count": counts.values,
        "Percentage": pct.values,
        'Label':  ['No Disease' if c == 0 else 'Heart Disease' for c in counts.index]        
    })
    display(imbalance_df)

    labels = ['No Disease', 'Heart Disease']
    colors = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.4, edgecolor='white')

    # add count + percentage on top of each bar
    for bar, count, pct in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f'{count}\n({pct}%)',
                ha='center', va='bottom', fontsize=11)

    ax.set_title('Class Distribution', fontsize=13, pad=12)
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.2)  
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()    


def plot_continuous_features(dataframe, target='num'):
    """
    Box plots for continuous features split by class.
    """
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    labels          = {0: 'No Disease', 1: 'Heart Disease'}
    colors          = {0: '#2ecc71',    1: '#e74c3c'}

    fig, axes = plt.subplots(1, len(continuous_cols), figsize=(18, 5))

    for ax, col in zip(axes, continuous_cols):
        data_by_class = [
            dataframe[dataframe[target] == cls][col].dropna()
            for cls in [0, 1]
        ]

        bp = ax.boxplot(data_by_class,
                        patch_artist=True,
                        widths=0.5,
                        medianprops=dict(color='white', linewidth=2))

        # color each box
        for patch, cls in zip(bp['boxes'], [0, 1]):
            patch.set_facecolor(colors[cls])

        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels.values(), fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Continuous Features by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()    

def plot_categorical_features(dataframe, target='num'):
    """
    Count plots for categorical and binary features split by class.
    """
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    binary_cols      = ['sex', 'fbs', 'exang']
    ordinal_cols     = ['ca']
    
    all_cols = categorical_cols + binary_cols + ordinal_cols
    
    labels  = {0: 'No Disease', 1: 'Heart Disease'}
    colors  = {0: '#2ecc71',    1: '#e74c3c'}
    
    n_cols  = 3
    n_rows  = int(np.ceil(len(all_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    
    for ax, col in zip(axes, all_cols):
        # count per value per class
        grouped = dataframe.groupby([col, target]).size().unstack(fill_value=0)
        
        x      = np.arange(len(grouped.index))
        width  = 0.35
        
        for i, cls in enumerate([0, 1]):
            if cls in grouped.columns:
                ax.bar(x + i * width,
                       grouped[cls],
                       width=width,
                       color=colors[cls],
                       label=labels[cls],
                       edgecolor='white')
        
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(grouped.index.astype(int), fontsize=10)
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    
    # hide any unused subplots
    for ax in axes[len(all_cols):]:
        ax.set_visible(False)
    
    fig.suptitle('Categorical & Binary Features by Class', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()   

def plot_correlation_heatmap(dataframe, target='num'):
    """
    Correlation heatmap of all features including target.
    """
    import seaborn as sns

    corr = dataframe.corr()

    # mask upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                ax=ax)

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()

    from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor():
    """
    Preprocessing pipeline for the Heart Disease dataset.
    - Continuous: median imputation + standard scaling
    - Categorical/ordinal: most_frequent imputation + ordinal encoding
    """
    continuous_cols  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['cp', 'restecg', 'slope', 'thal', 
                        'sex', 'fbs', 'exang', 'ca']

    numerical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    categorical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_preprocessor, continuous_cols),
        ('cat', categorical_preprocessor, categorical_cols)
    ])

    return preprocessor

def plot_pca(dataframe, target='num'):
    X = dataframe.drop(columns=[target]).copy()
    y = dataframe[target]

    # use the proper preprocessor instead of fillna
    preprocessor = build_preprocessor()
    X_processed  = preprocessor.fit_transform(X)

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = {0: '#2ecc71', 1: '#e74c3c'}
    labels  = {0: 'No Disease', 1: 'Heart Disease'}

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[cls], label=labels[cls],
                   alpha=0.6, edgecolors='white',
                   linewidths=0.5, s=60)

    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)', fontsize=11)
    ax.set_title('PCA — 2D Projection of Feature Space',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

    print(f"PC1 explains {var_explained[0]:.1f}% of variance")
    print(f"PC2 explains {var_explained[1]:.1f}% of variance")
    print(f"Total explained: {sum(var_explained):.1f}%")