#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

csv = '../../data/ai_io_requirements_2023_2026.csv'
figpath = '../figure_ai_io.png'

def main():
    df = pd.read_csv(csv)
    plt.figure(figsize=(8,4))
    df_train = df[df['WorkloadType']=='Training']
    df_inf = df[df['WorkloadType']=='Inference']
    plt.plot(df_train['Year'], df_train['Required_IOPS'], marker='o', label='Training IOPS')
    plt.plot(df_train['Year'], df_train['Required_Bandwidth_GBps'], marker='x', label='Training Bandwidth (GB/s)')
    plt.plot(df_inf['Year'], df_inf['Required_IOPS'], marker='s', label='Inference IOPS')
    plt.plot(df_inf['Year'], df_inf['Required_Bandwidth_GBps'], marker='^', label='Inference Bandwidth (GB/s)')
    plt.yscale('log')
    plt.xlabel('Year')
    plt.title('AI I/O Requirements (2023-2026) [Log Scale]')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    print('Saved', figpath)

if __name__ == '__main__':
    main()
