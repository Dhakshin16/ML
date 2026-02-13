# DNA Sequence Classification (Classical ML)

This project performs end-to-end DNA sequence classification using classical machine learning methods (no deep learning).

Contents
- `eda.py` - Exploratory Data Analysis and plots
- `feature_engineering.py` - k-mer extraction and feature preprocessing
- `train_models.py` - Train and evaluate Logistic Regression, SVM, Random Forest
- `utils.py` - helper functions for k-mer generation and data loading
- `requirements.txt` - Python dependencies
- `synthetic_dna_dataset.csv` - dataset (provided)

Run instructions

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run EDA:

```bash
python eda.py
```

3. Build features and train models:

```bash
python train_models.py
```

Project structure and explanations are included in the scripts as comments.
