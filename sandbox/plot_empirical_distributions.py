import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def plot_distribution(p):
    print(f'Plotting parameter {p}')
    csv_file = f'../external/JSFGermano2024/other_results/432192/src.tiv.RefractoryCellModel_JSF_6000/{p}_distribution.csv'
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 6))
    if p == 'lnV0':
        vals = np.log10(np.exp(df[p]))
    else:
        vals = df[p]
    plt.hist(vals, bins=30, edgecolor='black')
    plt.title(f'Distribution of {p}')
    plt.xlabel(p)
    plt.ylabel('Frequency')
    plt.savefig(f'outputs/{p}')




param_names = ['beta' ,'delta','lnV0','phi','pi','rho']
for p in param_names:
    plot_distribution(p)
