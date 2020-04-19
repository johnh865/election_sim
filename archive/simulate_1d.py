# -*- coding: utf-8 -*-

"""
Honest election simulations comparing IRV, plurality, and score voting




"""
import votesim
import votesim.simulation as sim
from votesim.votesystems import irv, plurality, score
from votesim.utilities.write import StringTable

import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd


#votesim.logconfig.setInfo()
#votesim.logconfig.setDebug()
votesim.logconfig.setWarning()


fname = 'simulation_output.txt'
trial_nums = 300
#tolerance = 2
cnum = 10
numwinners = 1
numcandidates = 4
seed = 0
build_plots = True
# Test 2 peaked, 1-dimensional. 


# Create the trial variations
size1 = 100
size2 = 100
width1 = .5
width2 = .5
#sizes2 = [0, 200, 400, 600, 800, 1000]
#widths1 = [.25, .5, 1]
#widths2 = [.25, .5, 1]


# Create the voters 
seed_start = 1
coords = [[-1.], [1.]]
#sizes = [1000, 2000]
#widths = [.5, .5]




# Create the candidates
candidate_pref_range = [-2, 2]
clen = candidate_pref_range[1] - candidate_pref_range[0]



#score_regrets = []
#irv_regrets = []
#plurality_regrets = []
candidates_record = []
parameters_record = []
regrets_record = []
for trial_no, seed in enumerate(range(seed_start, seed_start + trial_nums)):
    
    rs = votesim.randomstate.state
    rs.seed(seed)

#    size2 = rs.randint(0, 1000, 1)[0]
#    width1, width2  = rs.uniform(.25, 1, 2)
    sizes = [size1, size2]
    widths = [width1, width2]
    voters = sim.gaussian_preferences(coords, sizes, widths)
    tolerance = rs.uniform(.2, 4.0, 1)[0]
    voter_std = np.std(voters)
    voter_mean = np.mean(voters)    
    candidates = rs.uniform(-.5, .5, numcandidates) * voter_std * 4 + voter_mean
    candidates_record.append(np.sort(candidates))
    
    candidates1 = candidates[:, None]
    print('Trial #%d, Candidates=%s' % (trial_no, candidates))
    e = sim.ElectionRun(voters, candidates1,
                           numwinners=numwinners,
                           cnum=cnum,
                           tol=tolerance)
    
    desired_methods = ['irv',
                       'rrv',
                       'plurality',
                       'smith_minimax',
                       'star',
                       'ttr',
                       ]
    
    results = [e.run(method) for method in desired_methods]
    regrets = [e.stats.median_regret, results[0].ideal_regret] + \
              [r.consensus_regret for r in results]

    
    parameters = [seed, coords[0][0], size1, width1, coords[1][0], size2, width2,
                  tolerance, voter_std, voter_mean,]
    parameters_record.append(parameters)
    regrets_record.append(regrets)
    
    
    if build_plots and trial_no < 10:
        title = 'Trial #%d' % trial_no
        sim.plot1d(e, results, title)
        plt.savefig(title + '.png')
        plt.close()


regrets_record = np.array(regrets_record)
regrets_mean = np.mean(regrets_record, axis=0)
regrets_std = np.std(regrets_record, axis=0)



## write file
regrets_labels = ['Median_Regret', 'Ideal_Regret'] + \
                 [m + "_Regret" for m in desired_methods]

parameters_labels = ['Seed',
                     'Voters1_center', 'Voters1_size', 'Voters1_width',
                     'Voters2_center', 'Voters2_size', 'Voters2_width',
                     'Vote_tolerance', 'Voter_std', 'Voter_mean']
candidate_labels = ['Candidate#%d' % i for i in range(numcandidates)]
                    
parameters_record = np.array(parameters_record)
regrets_record = np.array(regrets_record)
candidates_record = np.array(candidates_record)

st = StringTable()
st.add(parameters_labels, parameters_record)
st.add(regrets_labels, regrets_record)
st.add(candidate_labels, candidates_record)
st.write(fname)

d1 = dict(zip(regrets_labels, regrets_record.T))
d2 = dict(zip(parameters_labels, parameters_record.T))
d3 = dict(zip(candidate_labels, candidates_record.T))
d = {}
d.update(d1)
d.update(d2)
d.update(d3)
df = pd.DataFrame(d)


