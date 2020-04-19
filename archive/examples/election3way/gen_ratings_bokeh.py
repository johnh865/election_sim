# -*- coding: utf-8 -*-
import numpy as np

import sys
import votesim
from votesim.votesystems import tools
from votesim.models import vcalcs, spatial

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import GnBu3, OrRd3, Category10
from bokeh.plotting import figure
from bokeh.layouts import gridplot

#############################################################################
### Create two simulations using 2 strategies
voternum = 1000
candidates = [-0.1, 0.2, 0.7]
candidates = np.atleast_2d(candidates).T


v1 = spatial.SimpleVoters(seed=0, strategy='candidate')
v1.add_random(voternum)
c1 = spatial.Candidates(voters=v1, seed=0)
c1.add(candidates)
v1.calc_ratings(c1.candidates)


v2 = spatial.SimpleVoters(seed=0, strategy='voter', stol=2.)
v2.add_random(voternum)
c2 = spatial.Candidates(voters=v2, seed=0)
c1.add(candidates)
v2.calc_ratings(c1.candidates)

#############################################################################
### Create histogram

hist, bins = np.histogram(v1.voters, bins=20, density=True)
bins = (bins[0:-1] + bins[1:])/2.

#############################################################################
### Create the plots 

output_file("stacked_split.html")

colors = Category10[10]
plots=[]

p = figure(width=750, 
           plot_height=250, 
           title='Ratings vs Regret for all Voters')
#p.line(v1.di)


regrets = []
ratings = []


for i in range(3):
    
    plots2 = []
    cnum = i + 1
    color = GnBu3[i]
    d1 = v1.voters.ravel()
    r1 = v1.ratings.T[i].ravel()
    
    d2 = v2.voters.ravel()
    r2 = v2.ratings.T[i].ravel()
    
    
    ratings.append(r2.mean())
    
    p = figure(width=500, 
               plot_height=250, 
               title='Ratings vs Preferences for Candidate %s' % cnum)
    p.line(bins, hist,
           legend_label='voter distr.',
           line_dash = [1, 2],
           alpha=1,
           color=colors[0])
    
    p.x(d1, r1,
        legend_label='rating by candidate',
        size=10,
        alpha=.6,
        color=colors[1])
    
    p.circle(d2, r2,
             legend_label='rating by voter pop.',
             size=4,
             alpha=.6,
             color=colors[2])
    
    p.circle(candidates[i], 0.5,
             legend_label='candidate location',
             size=12,
             alpha=1, color=colors[3])
    
    p.xaxis.axis_label = 'voter preferences'
    p.yaxis.axis_label = 'voter ratings of candidate %s' % (i+1)
    if i==2:
        p.legend.location = 'top_left'    
    
    plots2.append(p)
    
    
    
    p = figure(width=500, 
               plot_height=250, 
               title='Ratings vs Regret for Candidate %s' % cnum)  
    d1 = v1.distances.T[i].ravel()
    d2 = v2.distances.T[i].ravel()
    p.x(d1,
        r1,
        legend_label='rating by candidate',
        size=10,
        alpha=.6,
        color=colors[1])
    p.circle(d2, 
             r2,
             legend_label='rating by voter pop.',
             size=4,
             alpha=.6,
             color=colors[2])    
    p.xaxis.axis_label = 'voter regrets'
    p.yaxis.axis_label = 'voter ratings of candidate %s' % (i+1)
    p.legend.location = 'center_right'    
    
    plots2.append(p)    
    plots.append(plots2)

    regrets.append(d1.mean())
#############################################################################

candidates = ['#1','#2','#3']
text = ['%.2f' % s for s in ratings]

p1 = figure(width=500, 
           plot_height=250, 
           x_range=candidates,
           title='Candidate Mean Ratings')  
p1.vbar(x=candidates, top=ratings, width=.4, tags=text)

#labels = LabelSet(x=candidates, y=ratings, text=text, y_offset=8,
#                  text_font_size="8pt", text_color="#555555", text_align='center')
#


p2 = figure(width=500, 
           plot_height=250, 
           x_range=candidates,
           title='Candidate Mean Regret')  
p2.vbar(x=candidates, top=regrets, width=.4)
plots.append([p1, p2])

grid = gridplot(plots )
show(grid)
    
    
    
#    plt.subplot(3, 1, i+1)
#    plt.plot(bins, hist, label='voter distr.')
#    plt.plot(d1, r1, 'x', label='strat candidate')
#    plt.plot(d2, r2, '.', label='strat voter')
#    plt.plot(candidates[i], .5, 'o', label='candidate loc.')
#    plt.legend()
#    plt.ylabel('voter rating')
#    plt.grid()
#    
#plt.xlabel('voter regret')
