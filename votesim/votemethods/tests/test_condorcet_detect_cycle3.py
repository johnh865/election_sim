"""
Perform condorcet winner checks
"""
# -*- coding: utf-8 -*-
import re
from ast import literal_eval

import numpy as np
import votesim
from votesim.models import spatial
import votesim.votemethods as vs
from votesim.votemethods.condorcet import ranked_pairs, has_cycle
from votesim.votemethods.condcalcs import VoteMatrix, VotePairs
from votesim.votemethods.condcalcs import (condorcet_winners_check,
                                           condorcet_check_one)


a = """0	4	16	10	6	12	7	13	11	17	18	8	2	9	3	14	5	1	19	15
8	5	7	0	18	17	2	19	15	9	14	4	12	1	3	16	11	13	10	6
19	2	13	0	15	11	6	16	9	14	17	5	8	3	1	10	4	7	18	12
0	9	5	11	16	3	13	6	12	1	15	7	18	19	14	10	4	17	2	8
19	9	7	2	12	3	18	1	5	11	10	13	8	0	17	15	4	6	16	14
19	1	7	0	18	13	10	17	14	8	15	3	12	6	2	16	4	9	11	5
10	13	5	19	0	12	1	15	8	7	11	2	17	3	4	18	6	16	14	9
0	4	10	5	6	7	15	8	13	11	18	9	3	19	14	16	2	1	17	12
19	9	3	15	0	1	12	7	8	4	11	5	16	18	13	17	2	14	10	6
0	4	5	7	18	2	16	3	6	9	13	10	12	19	15	14	1	8	17	11
10	12	5	19	0	14	1	15	9	7	11	2	17	3	4	18	6	16	13	8
0	8	5	19	17	7	10	9	15	1	13	4	18	16	11	12	6	14	2	3
0	3	14	19	5	12	16	15	17	6	18	13	7	2	4	1	9	11	10	8
16	6	5	11	0	2	10	4	3	13	8	7	15	19	9	17	1	12	18	14
7	13	1	16	0	6	12	10	5	9	3	2	18	15	14	19	8	17	11	4
0	1	13	18	7	8	16	10	15	14	19	11	2	9	5	6	4	3	17	12
6	7	8	0	18	17	5	19	15	9	14	4	12	2	1	16	11	13	10	3
9	8	6	7	19	2	17	1	3	10	4	11	14	0	15	18	5	13	16	12
0	4	13	7	14	3	17	2	15	5	18	10	9	19	16	8	1	6	12	11
0	4	15	16	2	11	18	13	17	3	19	14	8	7	5	1	9	10	6	12
0	3	4	16	19	2	12	6	7	8	14	5	15	18	11	17	1	10	13	9
9	13	3	7	19	4	14	2	5	6	1	10	0	18	15	16	12	17	8	11
0	2	11	16	15	6	13	8	9	12	18	4	5	19	7	14	1	3	17	10
0	5	12	6	8	4	17	7	14	9	18	10	3	19	16	13	1	2	15	11
13	8	3	12	0	2	10	6	1	11	5	7	18	17	9	19	4	15	16	14
19	7	4	8	18	1	16	2	5	6	11	9	17	0	13	15	3	14	12	10
0	3	13	16	15	9	5	11	8	14	18	6	7	10	2	17	1	4	19	12
19	13	12	1	4	3	18	2	10	11	14	15	6	0	17	8	7	5	9	16
15	9	5	17	0	4	8	10	1	11	7	2	18	13	6	19	3	16	14	12
14	4	10	0	19	13	1	18	8	11	15	5	12	3	2	17	6	7	16	9
0	1	13	18	8	7	17	9	16	11	19	12	4	14	5	6	3	2	15	10
15	8	6	19	0	11	2	14	5	7	12	1	17	9	3	18	4	16	13	10
18	2	4	0	19	13	9	16	14	5	11	1	15	10	6	17	7	12	8	3
0	2	11	15	17	8	5	10	7	12	16	3	9	14	4	18	1	6	19	13
0	7	12	8	9	3	17	4	14	2	18	10	15	19	16	6	1	13	5	11
1	16	4	15	0	8	7	9	3	13	2	5	18	12	10	19	11	17	14	6
0	1	15	19	8	10	14	13	11	16	18	9	4	7	2	6	3	5	17	12
0	5	6	12	18	2	14	3	9	4	17	7	16	19	13	15	1	11	10	8
19	10	8	1	9	2	18	3	7	11	12	13	6	0	17	15	5	4	16	14
0	7	5	19	18	8	9	11	14	2	12	3	16	15	10	17	4	13	6	1
4	15	5	14	0	6	3	7	2	12	1	8	18	9	10	19	11	17	16	13
0	5	16	6	1	12	18	10	17	7	19	14	3	15	11	4	8	2	9	13
17	6	8	19	0	7	5	12	3	9	11	1	16	13	4	18	2	15	14	10
6	10	2	18	0	8	11	13	12	5	9	3	17	15	14	19	4	16	7	1
0	2	12	17	15	7	10	8	9	13	18	4	5	19	6	14	1	3	16	11
0	1	13	19	9	8	16	14	15	12	18	10	4	7	2	6	3	5	17	11
19	6	7	13	0	3	10	4	2	12	9	5	15	17	8	16	1	11	18	14
0	2	7	14	18	3	13	4	9	10	17	6	8	19	12	15	1	5	16	11
14	6	10	0	19	8	1	13	7	11	15	5	17	4	2	18	3	12	16	9
4	15	1	16	0	8	2	13	6	12	5	3	18	7	9	19	10	17	14	11
1	14	2	16	0	7	9	8	5	11	3	4	18	15	12	19	10	17	13	6
1	15	3	16	0	8	2	13	7	12	5	4	18	9	10	19	11	17	14	6
15	11	1	13	0	3	12	6	10	2	9	4	19	18	14	17	8	16	7	5
0	2	10	16	17	3	15	4	13	5	18	8	9	19	14	11	1	7	12	6
4	14	1	16	0	7	8	12	5	10	3	2	18	15	11	19	9	17	13	6
17	7	5	16	0	3	10	8	1	11	6	4	18	15	9	19	2	13	14	12
10	6	5	0	19	9	12	15	18	3	16	2	14	13	8	17	7	11	4	1
0	4	7	12	19	2	11	5	3	14	9	6	13	18	10	16	1	8	17	15
13	10	5	0	19	14	2	16	12	6	11	1	17	8	3	18	7	15	9	4
0	4	14	8	11	10	7	12	9	15	17	5	3	16	6	18	2	1	19	13
16	8	4	15	0	2	10	6	1	11	7	5	18	17	9	19	3	13	14	12
0	4	6	19	12	11	16	15	18	1	17	5	14	13	8	7	9	10	2	3
19	5	2	17	0	4	9	10	8	6	12	3	15	16	11	18	1	14	13	7
11	5	10	18	0	7	1	9	6	13	12	4	16	8	3	19	2	15	17	14
0	1	14	19	16	9	7	10	8	15	17	5	6	11	3	12	2	4	18	13
7	6	8	0	18	17	5	19	15	9	14	3	12	2	1	16	11	13	10	4
19	3	10	0	16	14	7	18	13	12	17	4	6	2	1	11	5	8	15	9
14	11	2	18	0	10	3	15	6	7	5	1	17	13	9	19	4	16	12	8
17	5	8	19	0	4	7	9	3	10	11	2	15	14	6	18	1	13	16	12
0	2	9	16	17	3	13	6	12	11	18	5	7	19	10	14	1	4	15	8
0	7	5	11	18	1	14	3	6	4	12	8	17	19	13	16	2	15	10	9
17	7	4	13	0	1	11	5	3	10	8	6	16	18	9	19	2	14	15	12
19	9	2	14	0	1	12	5	8	4	11	6	16	18	13	17	3	15	10	7
10	9	8	18	0	7	1	13	4	12	11	3	17	6	2	19	5	16	15	14
3	15	4	14	0	7	5	8	2	12	1	6	18	11	9	19	10	17	16	13
19	8	6	4	12	1	16	3	7	13	10	11	9	0	15	17	2	5	18	14
0	7	11	8	12	4	17	6	13	1	18	9	15	19	16	5	2	14	3	10
11	12	1	14	0	2	13	8	7	6	5	3	18	17	15	19	9	16	10	4
0	5	14	9	10	4	17	3	15	2	18	13	12	19	16	6	1	8	7	11
17	7	4	19	0	10	3	14	9	6	11	1	16	13	8	18	2	15	12	5
8	6	7	0	19	15	4	18	13	9	12	1	16	3	2	17	11	14	10	5
9	12	6	16	0	3	1	11	2	13	10	7	18	8	5	19	4	15	17	14
0	1	7	19	18	9	12	16	14	6	17	4	11	10	3	15	2	8	13	5
14	11	2	17	0	7	4	12	3	9	5	1	18	13	8	19	6	16	15	10
11	8	9	18	0	6	1	10	5	13	12	4	17	7	3	19	2	15	16	14
13	11	5	18	0	8	1	14	6	9	10	2	17	7	3	19	4	16	15	12
0	5	15	9	8	7	17	6	16	3	18	13	12	19	14	1	4	10	2	11
9	12	2	16	0	3	4	10	1	13	6	5	18	11	7	19	8	17	15	14
0	3	13	8	6	5	16	7	15	12	19	9	4	18	10	14	2	1	17	11
0	10	6	11	16	2	14	5	9	1	13	7	18	19	15	12	3	17	4	8
11	7	10	17	0	5	1	9	4	13	12	6	16	8	3	19	2	14	18	15
12	7	6	19	11	9	14	13	0	3	18	5	17	16	10	4	8	15	1	2
0	2	10	18	19	4	7	8	5	11	14	3	13	17	6	15	1	9	16	12
0	8	9	7	13	1	18	3	10	2	15	11	16	19	17	6	4	14	5	12
15	10	4	18	0	7	5	12	3	9	8	1	17	13	6	19	2	16	14	11
18	5	6	15	0	3	9	7	2	11	10	4	16	14	8	19	1	13	17	12
0	4	9	19	18	2	13	10	12	3	16	6	14	17	8	15	1	11	7	5
14	8	6	18	0	7	2	13	5	9	10	1	17	11	3	19	4	16	15	12
17	12	2	11	19	1	13	3	6	4	10	8	18	0	14	15	5	16	7	9
0	2	5	18	19	3	12	7	9	6	15	4	13	17	10	16	1	11	14	8
19	2	11	0	18	8	5	14	6	12	15	4	13	7	1	17	3	9	16	10
19	3	14	0	10	12	7	16	11	15	17	9	5	2	1	6	4	8	18	13
19	3	13	0	11	15	9	18	10	14	16	4	7	2	1	5	6	8	17	12
2	14	3	13	0	7	12	8	4	10	1	6	18	16	15	19	9	17	11	5
19	7	5	4	16	2	18	1	6	8	11	12	10	0	17	14	3	9	15	13
17	13	11	1	5	3	19	2	4	9	10	15	8	0	18	12	7	6	14	16
15	5	7	0	19	13	3	17	12	8	11	1	16	4	2	18	9	14	10	6
16	5	9	0	19	11	2	17	7	10	13	4	15	3	1	18	6	12	14	8
11	12	1	14	0	3	13	7	10	5	8	4	18	17	15	19	9	16	6	2
19	3	6	0	17	15	9	18	14	7	12	1	13	4	2	16	8	11	10	5
14	10	1	15	0	2	11	7	4	8	6	5	18	17	13	19	3	16	12	9
11	8	6	7	19	1	15	2	3	9	4	10	17	0	14	18	5	13	16	12
19	1	5	0	18	13	11	17	15	6	14	3	12	8	2	16	7	9	10	4
9	8	5	19	0	7	11	10	14	2	12	4	18	17	13	16	6	15	3	1
16	7	4	9	0	1	13	3	2	10	6	8	17	19	12	18	5	14	15	11
4	15	3	16	0	7	6	8	2	12	1	5	18	13	9	19	10	17	14	11
10	11	4	12	0	2	9	5	1	13	3	7	18	14	8	19	6	16	17	15
0	1	15	18	7	10	13	11	12	16	19	9	2	8	3	6	5	4	17	14
11	10	4	12	0	2	8	5	1	13	9	6	17	14	7	19	3	16	18	15
7	14	4	12	17	6	15	8	13	2	11	5	0	19	16	10	9	18	1	3
18	3	8	0	19	10	5	15	6	9	12	1	16	13	4	17	2	14	11	7
0	3	9	16	17	2	15	5	14	4	18	8	13	19	12	11	1	10	7	6
0	4	13	17	7	11	16	15	18	1	19	10	12	14	5	3	8	9	2	6
0	6	8	9	16	2	13	4	3	14	11	7	10	19	12	17	1	5	18	15
0	2	7	14	18	3	13	4	8	9	17	5	10	19	12	15	1	6	16	11
0	4	16	10	3	13	11	14	12	17	18	9	1	6	5	7	8	2	19	15
15	7	6	5	19	2	16	1	4	11	8	10	12	0	14	17	3	9	18	13
0	4	15	9	3	11	17	12	13	16	19	10	1	8	5	6	7	2	18	14
0	10	7	11	16	4	12	8	13	1	15	5	18	19	14	9	3	17	2	6
15	10	7	1	11	3	17	2	5	13	9	12	8	0	16	18	4	6	19	14
7	11	5	9	0	3	12	4	1	13	2	8	18	17	10	19	6	16	15	14
18	5	6	14	0	3	9	7	2	11	10	4	15	16	8	19	1	13	17	12
19	9	2	17	0	3	10	8	12	1	11	4	16	15	13	18	6	14	7	5
0	3	13	17	15	10	4	11	9	14	18	6	7	8	2	16	1	5	19	12
0	2	14	11	12	4	17	3	16	5	18	13	10	19	15	6	1	7	8	9
0	6	9	4	11	2	18	1	10	8	16	12	7	19	17	14	3	5	15	13
3	14	4	16	0	7	5	8	2	12	1	6	18	10	9	19	11	17	15	13
0	1	14	19	13	10	16	15	17	3	18	12	5	11	2	4	6	9	8	7
18	4	16	11	7	13	6	14	12	17	19	9	2	5	3	10	8	1	0	15
17	3	15	0	9	12	5	13	11	16	18	10	4	1	2	6	7	8	19	14
18	8	4	19	0	7	9	11	13	1	10	3	16	15	12	17	6	14	5	2
12	7	5	11	0	3	10	4	2	14	9	6	17	16	8	19	1	13	18	15
19	1	12	0	18	10	9	16	7	13	15	4	8	6	2	14	3	5	17	11
8	6	7	0	19	15	4	18	13	9	12	3	16	1	2	17	11	14	10	5
0	3	13	9	7	5	16	6	14	11	18	8	4	19	12	15	1	2	17	10
3	15	5	14	0	6	4	8	1	12	2	7	18	11	9	19	10	17	16	13
0	5	14	8	13	3	17	2	15	4	18	12	10	19	16	7	1	6	9	11
17	7	4	19	0	10	3	14	9	6	11	1	16	13	5	18	2	15	12	8
0	4	12	15	17	9	2	10	8	13	16	5	7	11	3	18	1	6	19	14
9	8	7	6	0	4	13	3	1	15	2	10	14	19	12	17	5	11	18	16
10	13	2	15	0	4	8	6	1	9	3	5	18	16	12	19	7	17	14	11
0	4	7	19	16	8	12	11	17	1	18	5	15	14	9	10	6	13	2	3
0	3	14	6	4	7	18	10	16	12	19	13	2	17	9	8	5	1	15	11
0	2	14	16	5	8	18	10	17	12	19	13	1	9	7	3	6	4	15	11
0	8	5	15	17	7	10	9	14	1	11	4	19	18	13	12	6	16	2	3
0	4	15	5	3	9	18	8	17	10	19	14	2	16	11	6	7	1	12	13
19	8	5	3	12	1	18	2	6	10	11	13	9	0	17	15	4	7	16	14
0	9	7	15	18	3	10	8	11	1	14	5	17	19	12	13	2	16	4	6
0	3	15	12	8	10	7	11	9	16	18	6	4	13	5	17	2	1	19	14
13	10	8	2	12	3	19	1	4	11	6	14	9	0	18	16	5	7	17	15
0	9	5	14	17	7	10	8	12	1	11	3	19	18	15	13	6	16	2	4
6	7	8	0	18	17	3	19	15	9	14	4	12	1	2	16	11	13	10	5
0	10	3	12	16	4	11	9	13	1	14	6	19	18	15	8	5	17	2	7
8	11	5	12	0	3	10	4	1	13	2	6	18	15	9	19	7	17	16	14
0	9	5	14	16	8	11	10	15	2	12	4	19	18	13	7	6	17	1	3
19	9	7	1	10	3	16	5	8	13	11	12	6	0	15	17	4	2	18	14
8	5	7	0	17	18	4	19	16	9	15	3	13	1	2	11	12	14	10	6
0	3	5	18	19	4	10	11	8	6	13	2	15	16	9	17	1	14	12	7
0	13	4	9	16	3	14	5	10	1	11	7	18	19	15	12	6	17	2	8
1	16	4	15	0	7	5	8	3	13	2	6	18	9	11	19	12	17	14	10
7	10	4	9	0	3	11	5	1	13	2	8	18	17	12	19	6	16	15	14
0	3	16	14	5	11	12	13	10	17	18	9	1	8	4	7	6	2	19	15
1	16	5	15	0	7	2	9	4	13	3	6	18	10	11	19	12	17	14	8
5	13	10	3	16	6	8	4	1	17	2	12	14	15	11	19	7	9	0	18
0	3	15	8	4	11	16	12	13	17	19	9	2	10	5	7	6	1	18	14
15	7	4	14	0	1	10	6	2	11	8	5	18	16	9	19	3	13	17	12
0	6	12	5	9	3	17	2	15	8	18	10	7	19	16	13	1	4	14	11
10	8	7	5	19	2	18	1	3	9	4	11	16	0	17	15	6	14	13	12
19	2	7	0	18	13	6	16	11	8	14	1	15	9	3	17	4	12	10	5
17	3	15	0	10	12	6	13	11	16	18	9	4	2	1	8	5	7	19	14
0	4	14	6	7	5	17	8	16	9	18	12	3	19	15	11	2	1	13	10
5	12	7	8	0	4	10	3	1	15	2	11	17	14	9	18	6	13	19	16
2	13	3	12	0	6	14	7	5	9	1	8	18	17	15	19	11	16	10	4
16	7	1	18	0	4	10	11	5	6	9	3	17	15	12	19	2	14	13	8
16	9	2	18	0	7	6	13	10	4	8	1	17	14	12	19	3	15	11	5
0	10	8	12	16	7	11	9	13	2	15	5	18	19	14	6	3	17	1	4
0	3	11	13	16	6	9	10	5	12	15	2	7	17	8	18	1	4	19	14
0	1	15	19	12	11	9	13	10	16	17	7	5	6	2	8	3	4	18	14
14	12	1	11	0	2	13	4	10	3	6	5	19	18	15	17	9	16	8	7
4	13	3	18	0	11	6	15	8	9	5	1	17	14	12	19	7	16	10	2
0	5	8	19	9	14	16	15	18	1	17	6	11	10	7	4	12	13	2	3
11	5	8	0	19	14	1	17	10	9	12	4	16	2	3	18	6	15	13	7
4	13	3	12	0	5	8	6	1	11	2	7	18	15	10	19	9	17	16	14
15	10	2	12	0	1	13	3	4	7	5	8	17	18	14	19	6	16	11	9
0	4	13	6	9	11	8	12	10	14	16	5	3	17	7	18	2	1	19	15
14	8	3	19	0	12	7	16	11	4	10	1	17	13	9	18	6	15	5	2
0	2	15	16	6	9	13	10	11	17	19	7	3	12	5	8	4	1	18	14
17	8	3	19	0	7	9	13	12	4	10	1	16	15	11	18	5	14	6	2
0	1	15	17	9	12	8	13	11	16	18	7	4	6	3	10	5	2	19	14
9	8	4	19	0	7	10	11	14	3	12	2	17	16	13	18	6	15	5	1"""
#
a1 = a.split('\n')
a2 = [ai.split() for ai in a1]
a = np.array(a2, dtype=int)

vm = VoteMatrix(a)
vp = VotePairs(vm.pairs)
vp1 = vp.prune_losers()

for i in range(18):
    vp1 = vp1.prune_losers()
    print(vp1.pairs.shape, 'losers=', vp1.condorcet_losers)

h = has_cycle(vm.pairs)
assert h == False

w, t, s = condorcet_winners_check(pairs=vm.pairs)
w2 = condorcet_check_one(a)
assert w2 in w