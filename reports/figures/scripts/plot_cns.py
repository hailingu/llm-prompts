#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

csv = '../../data/cloud_native_storage_adoption.csv'
figpath = '../figure_cns.png'

def main():
    df = pd.read_csv(csv)
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['CSI_Usage_Index'], marker='o', label='CSI Usage Index')
    plt.plot(df['Year'], df['Rook_AdoptionIndex'], marker='s', label='Rook Adoption Index')
    plt.plot(df['Year'], df['Longhorn_AdoptionIndex'], marker='^', label='Longhorn Adoption Index')
    plt.xlabel('Year')
    plt.ylabel('Index')
    plt.title('Cloud Native Storage Adoption (2023-2026)')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    print('Saved', figpath)

if __name__ == '__main__':
    main()
