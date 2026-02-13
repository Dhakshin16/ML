"""Build features: extract 3-mer frequencies and combine with GC/AT content."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import build_kmers, kmer_counts, load_data

def build_feature_matrix(df, k=3):
    kmers = build_kmers(k)
    kmer_features = np.array([kmer_counts(seq.upper(), k=k, kmers=kmers) for seq in df['Sequence']])
    # Combine with GC and AT content
    meta = df[['GC_Content','AT_Content']].to_numpy()
    X = np.hstack([kmer_features, meta])
    feature_names = kmers + ['GC_Content','AT_Content']
    return X, feature_names

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

if __name__ == '__main__':
    df = load_data()
    X, feature_names = build_feature_matrix(df, k=3)
    print('Feature matrix shape:', X.shape)
