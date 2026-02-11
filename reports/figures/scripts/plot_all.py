#!/usr/bin/env python3
import subprocess

scripts = [
    'plot_market.py',
    'plot_nvme.py',
    'plot_pmem.py',
    'plot_media_share.py',
    'plot_cns.py',
    'plot_ai_io.py'
]
for s in scripts:
    print('Running', s)
    subprocess.run(['python3', s], cwd='.')
print('All plots attempted.')
