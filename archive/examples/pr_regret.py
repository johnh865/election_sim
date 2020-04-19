# -*- coding: utf-8 -*-

import votesim
from votesim.metrics import PR_regret
import numpy as np



voters = [-1]*5 + [0]*5 + [1]*5
voters = np.atleast_2d(voters).T

# Candidates with same preferences as voters
winners = [-1, 0, 1]
winners = np.atleast_2d(winners).T
r, std = PR_regret(voters, winners)
print(r, std)



winners = [-1, 0,]
winners = np.atleast_2d(winners).T
r, std = PR_regret(voters, winners)
print(r, std)



winners = [0]
winners = np.atleast_2d(winners).T
r, std = PR_regret(voters, winners)
print(r, std)


winners = [-1,]
winners = np.atleast_2d(winners).T
r, std = PR_regret(voters, winners)
print(r, std)


