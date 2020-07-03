# -*- coding: utf-8 -*-
import numpy as np
import votesim
import votesim.metrics
#votesim.metrics.
#
#def median_regret(voters, weights=None):
#    num = len(voters)
#    center = np.median(voters, axis=0)
#    if weights is None:
#        diff = voters - center
#    else:
#        diff = (voters - center) * weights
#    dist = np.sum(np.linalg.norm(diff, axis=1)) / num
#    return dist
#
#
#def average_regret(voters, weights=None):
#    num = len(voters)
#    r = votesim.metrics.candidate_regrets(voters, voters)
#    
#    
#    
    

v = votesim.models.spatial.Voters(0)
v.add_random(5000, 2)

regret_mean = votesim.metrics.mean_regret(v.pref)
regret_median = votesim.metrics.median_regret(v.pref)
regret_std = votesim.metrics.regret_std(v.pref)
regret_voters = votesim.metrics.voter_regrets(v.pref)

print(regret_mean)
print(regret_median)
print(regret_std)
print(regret_voters.min())
print(regret_voters.mean())
print(np.median(regret_voters))

print(regret_mean + regret_std)
