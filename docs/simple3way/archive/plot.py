# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


import votesim
from votesim.benchmarks.simpleNd import SimpleThreeWay
dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, SimpleThreeWay.name)
def read():
    return SimpleThreeWay.read(dirname=dirname)


# Create the data
# rs = np.random.RandomState(1979)
# x = rs.randn(500)
# g = np.tile(list("ABCDEFGHIJ"), 50)
# df = pd.DataFrame(dict(x=x, g=g))
# m = df.g.map(ord)
# df["x"] += m


p = read()
df = p.dataframe
yname = 'args.etype'
xname = 'output.winner.regret_efficiency_voter'
ynew = 'type'
xnew = 'regret'

keys = [yname, xname]

df = df[keys]
df = df.rename(columns={yname : ynew, xname : xnew})


df[xnew] = 1-df[xnew]


# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row=ynew, hue=ynew, aspect=15, height=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, xnew, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, xnew, clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
    ax.set_xlim(None, 1)

g.map(label, "regret")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)