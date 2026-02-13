"""Exploratory Data Analysis for synthetic_dna_dataset.csv"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

sns.set(style='whitegrid')

def run_eda(path='synthetic_dna_dataset.csv'):
    df = load_data(path)
    print('Shape:', df.shape)
    print('Columns:', df.columns.tolist())
    print('\nMissing values:\n', df.isnull().sum())
    print('\nClass distribution:\n', df['Label'].value_counts())

    # Plots
    plt.figure(figsize=(8,4))
    sns.histplot(df['GC_Content'], kde=True, bins=30)
    plt.title('GC_Content distribution')
    plt.savefig('gc_content_dist.png')

    plt.figure(figsize=(8,4))
    sns.histplot(df['AT_Content'], kde=True, bins=30)
    plt.title('AT_Content distribution')
    plt.savefig('at_content_dist.png')

    plt.figure(figsize=(6,5))
    sns.heatmap(df[['GC_Content','AT_Content','Sequence_Length','Label']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation heatmap')
    plt.savefig('correlation_heatmap.png')

    print('Saved plots: gc_content_dist.png, at_content_dist.png, correlation_heatmap.png')

if __name__ == '__main__':
    run_eda()
