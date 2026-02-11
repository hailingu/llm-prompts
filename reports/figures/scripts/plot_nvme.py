#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

csv = '../../data/nvme_adoption_2023_2026.csv'
figpath = '../figure_nvme.png'

def main():
    df = pd.read_csv(csv)
    plt.figure(figsize=(8,4))
    plt.plot(df['Year'], df['NVMe_Server_Percent'], marker='o', label='NVMe Server (%)')
    plt.plot(df['Year'], df['NVMe_oF_Deployments_Percent'], marker='o', label='NVMe-oF Deployments (%)')
    plt.plot(df['Year'], df['Cloud_NVMe_Server_Percent'], marker='x', linestyle='--', label='Cloud NVMe Server (%)')
    plt.plot(df['Year'], df['Enterprise_NVMe_Server_Percent'], marker='x', linestyle='--', label='Enterprise NVMe Server (%)')
    plt.xlabel('Year')
    plt.ylabel('Percent')
    plt.title('NVMe / NVMe-oF Adoption (2023-2026)')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    print('Saved', figpath)

if __name__ == '__main__':
    main()
