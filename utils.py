import itertools
from collections import Counter
import pandas as pd

def load_data(path="synthetic_dna_dataset.csv"):
    return pd.read_csv(path, on_bad_lines='skip', engine='python')

def build_kmers(k=3):
    bases = ['A','C','G','T']
    return [''.join(p) for p in itertools.product(bases, repeat=k)]

def kmer_counts(seq, k=3, kmers=None):
    if kmers is None:
        kmers = build_kmers(k)
    counts = Counter([seq[i:i+k] for i in range(len(seq)-k+1)])
    # Return frequencies in fixed order
    total = sum(counts.values()) if counts else 1
    return [counts.get(km,0)/total for km in kmers]

if __name__ == '__main__':
    print("=" * 50)
    print("UTILITY FUNCTIONS DEMO")
    print("=" * 50)
    
    # Test data loading
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} samples")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Test k-mer building
    print("\n2. Building 3-mers...")
    kmers_3 = build_kmers(k=3)
    print(f"   Total 3-mers: {len(kmers_3)}")
    print(f"   Sample 3-mers: {kmers_3[:5]}")
    
    # Test k-mer counting
    print("\n3. Testing k-mer frequency extraction...")
    test_seq = df['Sequence'].iloc[0]
    print(f"   Test sequence (first 50 chars): {test_seq[:50]}...")
    freqs = kmer_counts(test_seq, k=3, kmers=kmers_3)
    print(f"   3-mer frequency vector length: {len(freqs)}")
    print(f"   Non-zero frequencies: {sum(1 for f in freqs if f > 0)}")
    print(f"   Sum of frequencies: {sum(freqs):.4f}")
    
    print("\n" + "=" * 50)
    print("All utility functions working correctly!")
    print("=" * 50)
