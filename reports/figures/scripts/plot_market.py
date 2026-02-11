#!/usr/bin/env python3
"""绘制市场规模趋势图（示例脚本）
依赖: pandas, matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('../../data/global_china_market_2023_2026.csv')
    piv = df.pivot(index='Year', columns='Region', values='MarketSize_USD_Bn')
    piv.plot(kind='bar')
    plt.title('Global vs China Storage Market (2023-2026)')
    plt.ylabel('Market Size (USD Billion)')
    plt.tight_layout()
    plt.savefig('../figure_market.png')
    print('Saved ../figure_market.png')
