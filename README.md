# MLCB 2026 — Assignment 2
## Heart Disease Classification using Repeated Nested Cross-Validation

### Overview
This repository contains the implementation of a repeated nested 
cross-validation (rnCV) pipeline for binary classification of 
coronary artery disease (CAD) using the UCI Cleveland Heart Disease 
dataset.

### Repository Structure
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory data analysis
│   ├── model_comparison.ipynb     # rnCV pipeline and results
│   └── error_analysis.ipynb       # Bonus error analysis
├── src/
│   ├── rncv.py                    # Repeated nested CV class
│   ├── preprocessing.py           # Preprocessing pipeline
│   └── functions.py               # Utility functions
├── models/
│   └── final_model.pkl            # Final deployment model
└── data/
└── heart.csv                  # Cleveland Heart Disease dataset


### Usage
```python
import pickle

# load final model
with open('models/final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# predict on raw input
predictions = model.predict(X_raw)
probabilities = model.predict_proba(X_raw)[:, 1]
```

### Results
- **Winner algorithm:** Linear Discriminant Analysis (LDA)
- **MCC:** 0.669 (95% CI: 0.642–0.679)
- **AUC:** 0.892 (95% CI: 0.883–0.908)
