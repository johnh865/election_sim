# -*- coding: utf-8 -*-

import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable
import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd

e = spatial.load_election('election3way.pkl')
e2 = e.rerun(index=3096)

dist = e2.voters.distances
scores = e2.ratings
ballots = e2.ballots / 5

for i in range(3):
    d = dist[:, i]
    s = scores[:, i]
    b = ballots[:, i]
    plt.figure()
    plt.plot(d, s, '.')
    plt.plot(d, b, '.')