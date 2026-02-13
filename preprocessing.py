"""Complete preprocessing pipeline for synthetic DNA dataset."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import load_data

def check_missing_values(df):
    """Check and report missing values."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])
        return True
    else:
        print("No missing values found.")
        return False

def handle_missing_values(df, strategy='drop'):
    """Handle missing values.
    
    Args:
        df: DataFrame
        strategy: 'drop' to remove rows, 'mean'/'median' to fill numeric columns
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy in ['mean', 'median']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            fill_value = df[col].mean() if strategy == 'mean' else df[col].median()
            df[col] = df[col].fillna(fill_value)
        return df
    return df

def remove_duplicates(df):
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    print(f"Removed {n_before - n_after} duplicate rows.")
    return df

def validate_sequences(df):
    """Validate DNA sequences contain only valid bases."""
    valid_bases = set('ACGTACGT')  # Case-insensitive
    invalid_rows = []
    
    for idx, seq in enumerate(df['Sequence']):
        if not all(c in valid_bases for c in str(seq).upper()):
            invalid_rows.append(idx)
    
    if invalid_rows:
        print(f"Found {len(invalid_rows)} sequences with invalid bases.")
        return df.drop(invalid_rows, errors='ignore')
    else:
        print("All sequences contain valid DNA bases.")
    
    return df

def normalize_sequences(df):
    """Convert all sequences to uppercase."""
    df['Sequence'] = df['Sequence'].str.upper()
    return df

def validate_numeric_ranges(df):
    """Validate numeric features are within expected ranges."""
    # GC_Content and AT_Content should be between 0 and 100 (percentages)
    issues = False
    
    if 'GC_Content' in df.columns:
        invalid = (df['GC_Content'] < 0) | (df['GC_Content'] > 100)
        if invalid.sum() > 0:
            print(f"Found {invalid.sum()} invalid GC_Content values (should be 0-100)")
            df = df[~invalid]
            issues = True
    
    if 'AT_Content' in df.columns:
        invalid = (df['AT_Content'] < 0) | (df['AT_Content'] > 100)
        if invalid.sum() > 0:
            print(f"Found {invalid.sum()} invalid AT_Content values (should be 0-100)")
            df = df[~invalid]
            issues = True
    
    if 'Sequence_Length' in df.columns:
        invalid = df['Sequence_Length'] <= 0
        if invalid.sum() > 0:
            print(f"Found {invalid.sum()} sequences with invalid length")
            df = df[~invalid]
            issues = True
    
    if not issues:
        print("All numeric features are within valid ranges.")
    
    return df

def encode_labels(df):
    """Encode categorical labels to numeric values."""
    if 'Label' in df.columns:
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'])
        print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        return df, le
    return df, None

def check_class_balance(df):
    """Check class distribution."""
    if 'Label' in df.columns:
        print("Class distribution:")
        print(df['Label'].value_counts())
        print("\nClass distribution (%):")
        print(df['Label'].value_counts(normalize=True) * 100)

def remove_outliers(df, method='iqr', threshold=1.5):
    """Remove outliers using IQR method.
    
    Args:
        df: DataFrame
        method: 'iqr' for interquartile range
        threshold: IQR multiplier (default 1.5)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    
    for col in numeric_cols:
        if col == 'Label':  # Skip label column
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
        if outliers.sum() > 0:
            print(f"Found {outliers.sum()} outliers in {col}")
            df_clean = df_clean[~outliers]
    
    print(f"Rows removed: {len(df) - len(df_clean)}")
    return df_clean

def full_preprocessing(path='synthetic_dna_dataset.csv', 
                       remove_outliers_flag=False,
                       encode_labels_flag=True):
    """Run complete preprocessing pipeline.
    
    Args:
        path: Path to input CSV file
        remove_outliers_flag: Whether to remove outliers
        encode_labels_flag: Whether to encode labels
    
    Returns:
        df_processed: Preprocessed DataFrame
        label_encoder: LabelEncoder instance (if encoding was applied)
    """
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(path)
    print(f"   Shape: {df.shape}")
    
    # Check missing values
    print("\n2. Checking missing values...")
    check_missing_values(df)
    
    # Remove duplicates
    print("\n3. Removing duplicates...")
    df = remove_duplicates(df)
    print(f"   Shape after removing duplicates: {df.shape}")
    
    # Normalize sequences
    print("\n4. Normalizing sequences (to uppercase)...")
    df = normalize_sequences(df)
    
    # Validate sequences
    print("\n5. Validating DNA sequences...")
    df = validate_sequences(df)
    print(f"   Shape after validation: {df.shape}")
    
    # Validate numeric ranges
    print("\n6. Validating numeric feature ranges...")
    df = validate_numeric_ranges(df)
    print(f"   Shape after validation: {df.shape}")
    
    # Remove outliers (optional)
    if remove_outliers_flag:
        print("\n7. Removing outliers...")
        df = remove_outliers(df, method='iqr', threshold=1.5)
    else:
        print("\n7. Skipping outlier removal.")
    
    # Check class balance
    print("\n8. Checking class balance...")
    check_class_balance(df)
    
    # Encode labels
    label_encoder = None
    if encode_labels_flag:
        print("\n9. Encoding labels...")
        df, label_encoder = encode_labels(df)
    else:
        print("\n9. Skipping label encoding.")
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df, label_encoder

if __name__ == '__main__':
    df_processed, le = full_preprocessing(remove_outliers_flag=False, encode_labels_flag=True)
    print("\nProcessed data preview:")
    print(df_processed.head())
    print("\nData types:")
    print(df_processed.dtypes)
    
    # Save preprocessed data
    df_processed.to_csv('preprocessed_dna_dataset.csv', index=False)
    print("\nPreprocessed data saved to preprocessed_dna_dataset.csv")
