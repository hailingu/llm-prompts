#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

csv = '../../data/pmem_scm_forecast_2023_2026.csv'
figpath = '../figure_pmem.png'

def main():
    df = pd.read_csv(csv)
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['SCM_Capacity_TB'], marker='o', label='SCM Capacity (TB)')
    plt.ylabel('Capacity (TB)')
    plt.xlabel('Year')
    ax2 = plt.twinx()
    ax2.plot(df['Year'], df['MarketSize_USD_M'], marker='x', color='orange', label='Market Size (USD M)')
    ax2.set_ylabel('Market Size (USD M)')
    plt.title('SCM / PMem Forecast (2023-2026)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath)
    print('Saved', figpath)

if __name__ == '__main__':
    main()
