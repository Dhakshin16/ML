"""Updated Train script: Integrates automated preprocessing with classical ML models."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import custom modules from the project
from preprocessing import full_preprocessing
from feature_engineering import build_feature_matrix, preprocess
from utils import load_data

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance using multiple metrics."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def main(path='synthetic_dna_dataset.csv'):
    # Step 1: Automated Preprocessing 
    # This handles missing values, normalization, and sequence validation
    df_processed, label_encoder = full_preprocessing(path=path)
    
    # Step 2: Feature Engineering [cite: 6]
    # Extracts 3-mer frequencies and combines them with GC/AT content
    X, feature_names = build_feature_matrix(df_processed, k=3)
    y = df_processed['Label'].to_numpy()

    # Step 3: Split and Scale Data [cite: 5]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s, scaler = preprocess(X_train, X_test)

    # Step 4: Model Training (Classical ML only) [cite: 3, 5]
    # Update the models list in train_models.py to test different kernels
    models = [
    ('LogisticRegression', LogisticRegression(max_iter=1000)),
    ('SVM_RBF', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
    ('SVM_Linear', SVC(kernel='linear', C=1.0, probability=True)),
    ('SVM_Poly', SVC(kernel='poly', degree=3, C=1.0, probability=True)),
    ('RandomForest', RandomForestClassifier(n_estimators=200, random_state=42))
    ]
    
    results = {}
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train_s, y_train)
        results[name] = evaluate_model(model, X_test_s, y_test)

    # Step 5: Report Results
    rows = []
    for name, res in results.items():
        rows.append([name, res['accuracy'], res['precision'], res['recall'], res['f1']])
    
    res_df = pd.DataFrame(rows, columns=['Model','Accuracy','Precision','Recall','F1'])
    print('\nModel Comparison:')
    print(res_df)

    # Display best model details
    best = res_df.sort_values('F1', ascending=False).iloc[0]['Model']
    print(f'\nBest Model (by F1): {best}')
    print('Confusion Matrix:')
    print(results[best]['confusion_matrix'])

if __name__ == '__main__':
    main()