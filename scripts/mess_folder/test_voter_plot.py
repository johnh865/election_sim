# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 1}
rc('font', **font)
sns.set()



p = r"D:\John Huang\Python\election_sim\data\benchmarks\simpleNd\simpleNd_vse_categories.csv"
df = pd.read_csv(p)
print(list(df.keys()))

##############################################################################
### Rename columns 

column_names = [
        'args.candidate.1.add_random.cnum', 
        'args.voter.1.add_random.ndim',
        'args.voter.0.init.strategy', 
        'args.election.1.run.etype',
        'stats.regret.efficiency_voter'    
    ]

new_names = [
    '# of Candidates',
    '# of Dimensions',
    'Voter Strategy',
    'Election Type',
    'VSE',
    ]
renamer = dict(zip(column_names, new_names))
df = df.rename(columns=renamer)


df['dissatisfaction'] = (1.0 - df['VSE'])*100
print(list(df.keys()))
##############################################################################
### Get average results
e_mean = df.groupby(['Election Type']).mean()
isort = np.argsort(e_mean['dissatisfaction'])
##############################################################################
### Group by Dimensions & Voter Strategy
### Plot 

args = [
        '# of Dimensions',
        'Voter Strategy',
        ]

groupby = df.groupby(args)
num_dims = len(np.unique(df['# of Dimensions']))
num_strats = len(np.unique(df['Voter Strategy']))

for ii, key in enumerate(groupby.groups):
    dfg = groupby.get_group(key)
    cnum = dfg['# of Candidates']
    etypes = dfg['Election Type']
    
    dfp = dfg.pivot('Election Type',
                    '# of Candidates',
                    'dissatisfaction'
                    )
    dfp = dfp.iloc[isort]
    
    ndim = key[0]
    strat = key[1]
    title = '# Dimensions=%s, Strategy=%s' % (ndim, strat)
    
    
    
    if ii % 2 == 1:
        yticklabels=False
    else:
        yticklabels=True
    
    if ii % 2 == 0:
        f, axes = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=(9, 4.5))
        axes = axes.ravel()
    ax = axes[ii % 2 ]
    
    sns.heatmap(dfp, 
                ax=ax,
                annot=True, 
                fmt=".1f", 
                cbar=False,
                linewidths=.5,
                yticklabels=yticklabels,
                vmax=25, 
                cmap='viridis_r',
                )
    if ii % 2 == 1:
        ax.set_ylabel('')
    ax.set_title(title)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.suptitle('Voter % Dissatisfaction ')
  

# Draw a heatmap with the numeric values in each cell
# f, ax = plt.subplots(figsize=(9, 6))
# sns.heatmap(dfp, annot=True, fmt="d", linewidths=.5, ax=ax)
        
        

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# # Load the example flights dataset and convert to long-form
# flights_long = sns.load_dataset("flights")
# flights = flights_long.pivot("month", "year", "passengers")

# # Draw a heatmap with the numeric values in each cell
# f, ax = plt.subplots(figsize=(9, 7))
# sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)