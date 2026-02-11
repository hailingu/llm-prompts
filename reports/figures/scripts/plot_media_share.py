#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

csv = '../../data/ssd_tape_hdd_share_2023_2026.csv'
figpath = '../figure_media_share.png'

def main():
    df = pd.read_csv(csv)
    years = df['Year']
    plt.figure(figsize=(8,4))
    plt.stackplot(years, df['SSD_Share'], df['HDD_Share'], df['Tape_Share'], labels=['SSD','HDD','Tape'])
    plt.legend(loc='upper left')
    plt.xlabel('Year')
    plt.ylabel('Market Share (%)')
    plt.title('SSD / HDD / Tape Market Share (2023-2026)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)
    print('Saved', figpath)

if __name__ == '__main__':
    main()
