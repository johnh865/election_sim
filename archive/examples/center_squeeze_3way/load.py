# -*- coding: utf-8 -*-

import votesim
from votesim.models import spatial
from votesim.utilities.write import StringTable
import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np
import pandas as pd

v = spatial.SimpleVoters(0)
v.add(1)
c = spatial.Candidates(v, 0)
e = spatial.Election(None, None)
e.voters = v
e.candidates = c
e.load_json('center_squeeze_part2_results.json')
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