# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from itertools import cycle

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import GnBu3, OrRd3, Category10
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column

import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable



class Bplot(object):
    def __init__(self, figure, spacing=10, line=True, marker=True):
        types = ['asterisk',
                      'circle',
                      'cross',
                      'diamond',
                      'hex',
                      'inverted_triangle',
                      'square',
                      'triangle',
                      'x']
        
        self._iter = cycle(types)
        self._figure = figure
        self._colors = cycle(Category10[10])
        self._spacing = spacing
        self._line = line
        self._marker = marker
        return
    
    
    def _plot_marker(self, *args, **kwargs):
        typei = next(self._iter)
        func = getattr(self._figure, typei)
        return func(*args, **kwargs)
        
    
    def plot(self, x, y, name='', size=6, line=None, marker=None, *args, **kwargs):
        p = self._figure
        color = next(self._colors)
        
        if line is None:
            line = self._line
        if marker is None:
            marker = self._marker
        
        if line:
            p.line(x=x, 
                   y=y, 
                   color=color,
                   alpha=.5,
                   line_width=1,
                   legend_label=name)
            
        if marker:
            self._plot_marker(x=x[:: self._spacing],
                              y=y[::self._spacing],
                              color=color,
                              size=size,
                              fill_alpha=0.0,
                              line_alpha=0.5,
                              legend_label=name)
        return
    
        
##################################################################################

e = spatial.load_election('election3way.pkl')
metric_name = 'stats.regret.efficiency_voter'
df0 = e.dataframe()
output_file("post.html")

##################################################################################

p0 = figure(width=800, plot_height=400, title='Candidate Locations for Elections')
groupby = df0.groupby('args.election.2.run.etype')
dfi = groupby.get_group('irv')
candidates1 = dfi['args.candidate.0.add.candidates']
candidates1 = np.column_stack(candidates1)

vnum = candidates1.shape[1]
trials = np.arange(vnum)
percentile = trials / vnum

plotter = Bplot(p0, line=False, spacing=1)
plotter.plot(percentile, candidates1[0], name='Left')
plotter.plot(percentile, candidates1[1], name='Center')
plotter.plot(percentile, candidates1[2], name='Right')
p0.legend.location = 'top_left'
p0.xaxis.axis_label = 'Percentage of Trials'
p0.yaxis.axis_label = 'Candidate Preference'

##################################################################################
### Get voter satisfactoins

df = df0.sort_values(by=metric_name)
groupby = df.groupby('args.election.2.run.etype')
names = list(groupby.groups.keys())

vse_table = []
for name in names:
    dfi = groupby.get_group(name)
    vsei = dfi[metric_name]
    vse_table.append(vsei)
vse_table = np.array(vse_table, dtype=float)

### Construct histogram
voterhist, hedges = np.histogram(e.voters.voters, bins=20, density=True)
ch_edges = .5 * (hedges[0:-1] + hedges[1:])

##################################################################################

p = figure(width=800, 
           plot_height=400, 
           y_range=[0, 1.1],
           title='Voter Satisfactions for all Elections')
plotter = Bplot(p)

for i, name in enumerate(names):
    vsei = vse_table[i]
    plotter.plot(percentile, vsei, name=name)

p.xaxis.axis_label = 'Percentage of Trials'
p.yaxis.axis_label = 'Voter Satisfaction (normalized by median regret)'
p.legend.location = 'bottom_right'

##################################################################################
### Get election results for each election type

plots = []
#groupby = df0.groupby('args.election.2.run.etype')
colors = Category10[10]
for name in names:
    dfi = groupby.get_group(name)
    winnersi = dfi['stats.winner.all']
    winnersi = np.row_stack(winnersi).ravel()
    besti = dfi['stats.candidate.best']
    besti = np.row_stack(besti).ravel()

    candidatesi = dfi['args.candidate.0.add.candidates']
    candidatesi = np.column_stack(candidatesi)
    winlocs = candidatesi[(winnersi, trials)]
    bestlocs = candidatesi[(besti, trials)]
    
    imatch = winlocs != bestlocs
    winlocs1 = winlocs[imatch]
    percentile1 = percentile[imatch]
    
    
    pi = figure(width=800,
                plot_height=400, 
                y_range=[-3, 3],
                title='Candidate Preference Locations for %s' % name)
    
    pi.line(voterhist, ch_edges, legend_label='Voter Distr.', color='black')
    pi.line([0, 1], [0, 0], color='black', alpha=.75)
    

    pi.cross(percentile, candidatesi[0], 
             color=colors[0], alpha=0.5, legend_label='Left', size=4)

    pi.cross(percentile, candidatesi[1],
             color=colors[1], alpha=0.5, legend_label='Center', size=4)

    pi.cross(percentile, candidatesi[2], 
             color=colors[2], alpha=0.5, legend_label='Right', size=4)
    
    pi.circle(percentile1, winlocs1, 
              color=colors[3], alpha=0.75, fill_alpha=0.0,
              legend_label='Missed Winner', size=8)
    
    pi.diamond(percentile, bestlocs, 
              color=colors[4], alpha=0.75, fill_alpha=0.0, 
              legend_label='Best', size=6)
    
    pi.line(voterhist, ch_edges, 
            color='black',
            legend_label='Voter Distr.',)

#    plotter.plot(percentile, winlocs, name='Winner', size=6)
#    plotter.plot(percentile, bestlocs, name='Best', size=6)
#    plotter.plot(percentile, candidatesi[0], name='Left', size=4)
#    plotter.plot(percentile, candidatesi[1], name='Center', size=4)
#    plotter.plot(percentile, candidatesi[2], name='Right', size=4)

#    plotter.plot(voterhist, ch_edges, name='Voter Distr.',
#                 marker=False, line=True)
    
    pi.legend.orientation = "horizontal"
    pi.xaxis.axis_label = 'Percentage of Trials, sorted by Satisfaction'
    pi.yaxis.axis_label = 'Candidate & Winner Preferences'


    plots.append(pi)

grid = column([p0, p] + plots)
show(grid)



#vse = df1[metric_name]
#
#vse_table = [vse[df1['args.election.2.run.etype'] == t] for t in types]
#vse_table = np.array(vse_table)
#
#ii_vse_sorted = np.argsort(vse_table, axis=1)
#
#typenum = len(types)
#for jj in range(typenum):
#    vse_table[jj] = vse_table[jj, ii_vse_sorted[jj]]
#
#
#### Get voter histogram
#voterhist, hedges = np.histogram(v.voters, bins=20, density=True)
#ch_edges = .5 * (hedges[0:-1] + hedges[1:])
#
##
##
#candidates1 = df['args.candidate.0.add.candidates']
#candidates1 = candidates1[df['args.election.2.run.etype'] == 'irv']
#candidates1 = np.column_stack(candidates1)
#
#
#winners = df['stats.winners']
#
#
#trials = np.arange(len(candidates1.T))
#epercent = trials / np.max(trials) * 100
#
#
#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(trials, candidates1[0], '.', label='Left')
#plt.plot(trials, candidates1[1], '.', label='Center')
#plt.plot(trials, candidates1[2], '.', label='Right')
#plt.xlabel('Trial No')
#plt.ylabel('Candidate location')
#plt.grid()
#plt.legend()
#
#xticks = np.arange(0, 101, 1)
#
#plt.subplot(2,1,2)
#for vsei, typei in zip(vse_table, types):
#    plt.plot(epercent, vsei, label=typei)
#plt.legend()
#plt.grid()
#plt.ylim(0, None)    
#plt.axhline(0)
#
#for jj, etype in enumerate(types):
#    plt.figure()
#    candidatesi = candidates1[:, ii_vse_sorted[jj]]
#    winnersi = winners[df['args.election.2.run.etype'] == etype]
#    winnersi = np.column_stack(winnersi).ravel()[ii_vse_sorted[jj]]
#    winlocs = candidatesi[(winnersi, trials)]
#    
#    
#    ax = plt.subplot(2,1,1)
#    plt.plot(epercent, candidatesi[0], '.', label='Left')
#    plt.plot(epercent, candidatesi[1], '.', label='Center')
#    plt.plot(epercent, candidatesi[2], '.', label='Right')
#    plt.plot(epercent, winlocs, 'o', alpha=.3, fillstyle='none', label='winner')
#    
#    
#    
#    plt.plot(voterhist*100, ch_edges, label='voters')
#    
#    plt.xlabel('Percentile of Trials')
#    plt.ylabel('Candidate location')
#    ax.set_xticks(xticks, minor=True)
#    plt.xticks
#    plt.axhline(0)
#    plt.grid(b=True, which='both')
#    plt.legend(ncol=3)
#    
#    ax = plt.subplot(2,1,2)
#    ax.set_xticks(xticks, minor=True)
#    
#    plt.plot(epercent, vse_table[jj], label=etype)
#    plt.xlabel('Percentile of Trials')
#    plt.ylabel('Voter Satisfaction**')    
#    plt.legend()
#    plt.grid(b=True, which='both')
#    plt.ylim(0, 1.1)    



