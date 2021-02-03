# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:29:42 2020

@author: John

Test TacticCompare to make sure the regret output makes sense. 
"""
import pytest
import numpy as np

import votesim
from votesim.models import spatial
from votesim.strategy import TacticalBallots
from votesim.metrics.groups import TacticCompare
from votesim import metrics

def test_metrics_compare():
    """Test plurality bullet voting strategy.
    
    Scenario attributes:
     - 3 candidate race with 3 concentrated, coincident voter groups
     
     >>> Candidates               #0   #1   #2
     >>> Preference locations =  [-1,  0.5,  1]
     >>> Number of voters =      [ 7,    3,  5]
     
     - If voters are honest, plurality candidate #0 should win with 7 votes 
     - If voters are strategic and consider only the top two candidates, 
       Candidate #1 supporters also lean towards #2 - therefore 
       candidate #2 should win with 3 + 5 = 8 votes.
       
    """
    pref = [-1]*7 + [0.5]*3 + [1]*5
    pref = np.array(pref)[:, None]
    v = spatial.Voters(seed=0)
    v.add(pref)
    
    cpref = [-1, .5, 1]
    cpref = np.array(cpref)[:, None]
    c = spatial.Candidates(v)
    c.add(cpref)
    
    #####################################################################
    # Run honest election    
    
    e1 = spatial.Election(voters=v, candidates=c)
    e1.run('plurality')
    result1 = e1.result
    estat1 = result1.stats
    #####################################################################
    # Run strategic election
  
    strategy1 = {'tactics' : 'bullet_preferred',
                 'subset' : '',
                 'ratio' : 1,
                 'underdog' : None}
    s = spatial.Strategies(v)
    s.add(**strategy1)
    
    
    e1.set_models(strategies=s)
    e1.run('plurality', result=result1)
    result2 = e1.result
    estat2 = result2.stats
    
    #####################################################################
    # Run the assestions
    
    tc = votesim.metrics.TacticCompare(estat2, estat1) 
    t = e1.ballotgen.tacticalballots
    topdog_num = len(t.group_index['topdog-0'])
    underdog_num = len(t.group_index['underdog-0'])
    
    regret1 = estat1.winner.regret
    regret2 = estat2.winner.regret
    regret_change = tc.regret
    
    # Make sure regret change adds up for tactical and honest voters
    assert regret2 - regret1 == regret_change['tactical-0']
    
    
    # Make sure group regrets add up for the total reget, honest. 
    regret_honest = tc._group_honest.regret
    regret1a = ((regret_honest['topdog-0'] * topdog_num
               + regret_honest['underdog-0'] * underdog_num) /
               (topdog_num + underdog_num))
    assert regret1a == regret1
    
    
    # Make sure group regrets add up for the total reget, tactical. 
    regret_strate = tc._group_strate.regret
    regret2a = ((regret_strate['topdog-0'] * topdog_num
               + regret_strate['underdog-0'] * underdog_num) /
               (topdog_num + underdog_num))
    assert regret2a == regret2
        
    return e1, estat2, estat1


if __name__ == '__main__':
    e, estat2, estat1 = test_metrics_compare()

    
    
    
    # result = e.result
    # dict1 = result.output
    # regret = dict1['output.winner.regret']

    # topdog_regret = dict1['output.tactic_compare.regret.tactical-topdog-0']
    # underdog_regret = dict1['output.tactic_compare.regret.tactical-underdog-0']
    # avg_regret = ((topdog_regret * topdog_num + underdog_regret * underdog_num) 
    #               / (topdog_num + underdog_num))
    

    
    