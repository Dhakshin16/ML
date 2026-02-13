"""
Generate Synthetic DNA Dataset for Classification
==================================================
Creates a realistic synthetic DNA dataset with multiple classes
based on different sequence characteristics.
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_dna_sequence(length, gc_bias=0.5):
    """
    Generate a random DNA sequence with specified GC content bias.
    
    Parameters:
    -----------
    length : int
        Length of the DNA sequence
    gc_bias : float
        Probability of GC nucleotides (0.0 to 1.0)
    
    Returns:
    --------
    str : DNA sequence
    """
    nucleotides = ['A', 'T', 'G', 'C']
    
    # Adjust probabilities based on GC bias
    gc_prob = gc_bias / 2  # Split between G and C
    at_prob = (1 - gc_bias) / 2  # Split between A and T
    
    probabilities = [at_prob, at_prob, gc_prob, gc_prob]
    
    sequence = ''.join(np.random.choice(nucleotides, size=length, p=probabilities))
    return sequence

def calculate_gc_content(sequence):
    """Calculate GC content percentage"""
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def calculate_at_content(sequence):
    """Calculate AT content percentage"""
    at_count = sequence.count('A') + sequence.count('T')
    return (at_count / len(sequence)) * 100

def generate_class_0_sequences(n_samples, seq_length=100):
    """
    Class 0: GC-rich sequences (50-70% GC)
    Represents gene-rich regions or promoters
    """
    sequences = []
    for _ in range(n_samples):
        gc_bias = np.random.uniform(0.50, 0.70)
        seq = generate_dna_sequence(seq_length, gc_bias)
        sequences.append(seq)
    return sequences

def generate_class_1_sequences(n_samples, seq_length=100):
    """
    Class 1: AT-rich sequences (60-80% AT)
    Represents intergenic regions or heterochromatin
    """
    sequences = []
    for _ in range(n_samples):
        gc_bias = np.random.uniform(0.20, 0.40)  # Low GC = High AT
        seq = generate_dna_sequence(seq_length, gc_bias)
        sequences.append(seq)
    return sequences

def generate_class_2_sequences(n_samples, seq_length=100):
    """
    Class 2: Balanced sequences (45-55% GC)
    Represents typical coding sequences
    """
    sequences = []
    for _ in range(n_samples):
        gc_bias = np.random.uniform(0.45, 0.55)
        seq = generate_dna_sequence(seq_length, gc_bias)
        sequences.append(seq)
    return sequences

def add_motifs(sequence, motif, positions):
    """
    Add specific motifs to sequences to make classes more distinguishable
    
    Parameters:
    -----------
    sequence : str
        Original DNA sequence
    motif : str
        Motif to insert
    positions : list
        Positions where to insert the motif
    
    Returns:
    --------
    str : Modified sequence
    """
    seq_list = list(sequence)
    for pos in positions:
        if pos + len(motif) <= len(seq_list):
            seq_list[pos:pos+len(motif)] = list(motif)
    return ''.join(seq_list)

def generate_dataset(n_samples_per_class=1000, seq_length=100):
    """
    Generate complete synthetic DNA dataset
    
    Parameters:
    -----------
    n_samples_per_class : int
        Number of samples for each class
    seq_length : int
        Length of each DNA sequence
    
    Returns:
    --------
    pd.DataFrame : Complete dataset
    """
    print("=" * 80)
    print("GENERATING SYNTHETIC DNA DATASET")
    print("=" * 80)
    
    all_sequences = []
    all_labels = []
    
    # Generate Class 0 (GC-rich)
    print(f"\n✓ Generating Class 0: GC-rich sequences ({n_samples_per_class} samples)")
    class_0_seqs = generate_class_0_sequences(n_samples_per_class, seq_length)
    
    # Add characteristic motifs to some Class 0 sequences
    for i, seq in enumerate(class_0_seqs):
        if i % 3 == 0:  # Add motif to 1/3 of sequences
            positions = [random.randint(0, seq_length - 10) for _ in range(2)]
            seq = add_motifs(seq, "GCGCGC", positions)
            class_0_seqs[i] = seq
    
    all_sequences.extend(class_0_seqs)
    all_labels.extend([0] * n_samples_per_class)
    
    # Generate Class 1 (AT-rich)
    print(f"✓ Generating Class 1: AT-rich sequences ({n_samples_per_class} samples)")
    class_1_seqs = generate_class_1_sequences(n_samples_per_class, seq_length)
    
    # Add characteristic motifs to some Class 1 sequences
    for i, seq in enumerate(class_1_seqs):
        if i % 3 == 0:
            positions = [random.randint(0, seq_length - 10) for _ in range(2)]
            seq = add_motifs(seq, "ATATAT", positions)
            class_1_seqs[i] = seq
    
    all_sequences.extend(class_1_seqs)
    all_labels.extend([1] * n_samples_per_class)
    
    # Generate Class 2 (Balanced)
    print(f"✓ Generating Class 2: Balanced sequences ({n_samples_per_class} samples)")
    class_2_seqs = generate_class_2_sequences(n_samples_per_class, seq_length)
    
    # Add characteristic motifs to some Class 2 sequences
    for i, seq in enumerate(class_2_seqs):
        if i % 3 == 0:
            positions = [random.randint(0, seq_length - 10) for _ in range(2)]
            seq = add_motifs(seq, "ATGCATGC", positions)
            class_2_seqs[i] = seq
    
    all_sequences.extend(class_2_seqs)
    all_labels.extend([2] * n_samples_per_class)
    
    # Calculate features
    print("\n✓ Calculating sequence features...")
    gc_contents = [calculate_gc_content(seq) for seq in all_sequences]
    at_contents = [calculate_at_content(seq) for seq in all_sequences]
    seq_lengths = [len(seq) for seq in all_sequences]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sequence': all_sequences,
        'GC_Content': gc_contents,
        'AT_Content': at_contents,
        'Sequence_Length': seq_lengths,
        'Label': all_labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  Total samples: {len(df)}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Number of classes: {len(df['Label'].unique())}")
    print(f"\nClass distribution:")
    print(df['Label'].value_counts().sort_index())
    
    return df

def save_dataset(df, filename='synthetic_dna_dataset.csv'):
    """Save dataset to CSV file"""
    df.to_csv(filename, index=False)
    print(f"\n✓ Dataset saved to '{filename}'")
    print("=" * 80)

# Main execution
if __name__ == "__main__":
    # Generate dataset with 1000 samples per class (3000 total)
    dataset = generate_dataset(n_samples_per_class=1000, seq_length=100)
    
    # Display sample data
    print("\nSample data (first 5 rows):")
    print(dataset.head())
    
    print("\nDataset statistics:")
    print(dataset.describe())
    
    # Save to CSV
    save_dataset(dataset)
     
    print("\n🎉 Dataset generation complete!")
    print("You can now run the main classification pipeline.")