#!/usr/bin/env python3
"""Sample size calc for mean difference (two-sided t-test)"""
import math
import argparse
from statsmodels.stats.power import TTestIndPower


def calc_sample_size(delta=0.3, sigma=1.0, alpha=0.05, power=0.8):
    # delta: minimal detectable effect (difference in means)
    # assume equal groups
    effect_size = delta / sigma
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative='two-sided')
    return math.ceil(n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.3, help='MDE for MOS (mean diff)')
    parser.add_argument('--sigma', type=float, default=1.0, help='std dev of MOS')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--power', type=float, default=0.8)

    args = parser.parse_args()
    n = calc_sample_size(delta=args.delta, sigma=args.sigma, alpha=args.alpha, power=args.power)
    print(f'Required samples per group: {n}')
