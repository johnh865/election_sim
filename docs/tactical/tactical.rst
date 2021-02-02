Strategic Voter Simulations
===========================================================================================

John Huang, 2 February 2021


Executive Summary
------------------

This report documents results for strategic voting simulations for a 
variety of proposed and popular voting methods. Assessed systems include:

 - `Approval voting <https://en.wikipedia.org/wiki/Approval_voting>`_
 - `Instant runoff voting (irv) <https://en.wikipedia.org/wiki/Instant-runoff_voting>`_
 - `Scored voting <https://en.wikipedia.org/wiki/Score_voting>`_
 - `STAR voting <https://en.wikipedia.org/wiki/STAR_voting>`_
 - `Ranked pairs <https://en.wikipedia.org/wiki/Ranked_pairs>`_
 - `Smith minimax <https://electowiki.org/wiki/Smith//Minimax>`_
 - `Smith score <https://electowiki.org/wiki/Smith//Score>`_
 
These new systems are compared to the traditional 
`plurality voting <https://en.wikipedia.org/wiki/Plurality_(voting)>`_ system.


Simulation output is shown below for 3 voter strategies - honest, one-sided, and two-sided 
strategy. `Voter Satisfaction Efficiency (VSE) <https://electionscience.github.io/vse-sim/VSE/>`_ 
is plotted for simulated elections
at the 5th percentile of lowest VSE, the 95th percentile of highest VSE, and the average VSE. 

.. figure:: Tactical-VSE.png
    
    Figure 1: Voter satisfaction efficiency for honest, one-sided, and two-sided strategy
	
	
	

Strategy Model
--------------
Voter strategy was devised upon predicting a winner by simulating an honest
election, and then performing coordinated voting ballot tactics on an "underdog" candidate in an effort
for the underdog candidate to win against the "topdog" honest winner. 


Honest Strategy
+++++++++++++++
An honest strategy in this simulation is defined to be a voter behavior where 
scores or ranks are constructed monotonically and proportionately to 
voter preference distance from a candidate. Honest strategy produces an "honest winner".


Front-Runners
+++++++++++++
Two candidates are chosen by voters to apply strategy. The topdog front-runner is always the honest winner. 
The underdog front-runner is chosen iteratively, however strategic voters will always coordinate their votes
in favor of this arbitrary front-runner. 

Tactics
+++++++
Tactics are ballot manipulation routines voters may use to boost
support for their preferred candidate. 

- Bury -- Rate or rank a front-runner the worst rank or rating equal to a voter's true most hated candidate.
- Deep Bury -- Rank a front-runner the worst rank, below a voter's true most hated candidate. 
- Compromise -- Give a front-runner the best rank or rating. 
- Truncate Hated -- All candidates equal or worse than the voter's hated front-runner are given the lowest ranking or score. 
- Truncate Preferred -- All candidates worse than the voter's favorite front-runner are given the lowest ranking or score. 
- Bullet Preferred -- The preferred front-runner is given the best score or rank. All other candidates are given the lowest rank/score. 
- Minmax Hated -- Give worst rating or ranking to candidates equal or worse than a voter's hated front-runner.
- Minmax Preferred -- Give worst rating or ranking to candidates worse than a voter's favorite front-runner.  
 

One-Sided Strategy
++++++++++++++++++
One sided strategy is defined to be a strategy where only supporters 
of an underdog candidate use *tactics*, while topdog supporters use honest 
strategy. 

The goal of the assessment is calculate potential worst-case tactical scenarios, assuming rational underdog voters
with perfect information. 
Given the enormous space of possible voter behavior, simulation post-processing
chooses a one-sided tactic that must increase the satisfaction of underdog strategic voters 
but also minimize average voter satisfaction. Tactics are iterated over all possible 
underdog candidates and test against all possible underdog-against-topdog voting coalitions. 
Every listed tactic is iteratively tested. All underdog coalition members use the same tactic, 
though their ballot markings may still be different from one-another. 

If all tactics backfire against the underdog (ie, result in reduced satisfaction for all underdog coalitions), 
then the honest election results are used. In other words, this analysis assumes 
that one-sided strategic voters are rational enough to avoid backfire. 


Two-Sided Strategy
++++++++++++++++++
In two-sided strategy, underdog voters use the same strategy as in one-sided strategy.
Topdog coalition members attempt to counter underdog strategy by bullet voting in favor of the honest winner. 

In contrast to one-sided strategy, two-sided simulations do not filter out backfired two-sided tactics. 
Two-sided strategy tests a voting method's ability to resist any sort of underdog manipulation. 

For the scope of this assessment, no other topdog counter strategies were considered. 


Simulation Details
------------------
- 6000 total voter/candidate distribution combinations were simulated.
- There are 51 voters for each election.
- Spatial preference dimensions of 1, 2, and 3 were used. 
- Voter preferences are normally distributed. 
- Candidate preferences are uniformly distributed +/-1.5 std deviations from the voter preference mean. 
- There are either 3 or 5 candidates in each election. 



Results
-------
A simple performance metric is devised as the average of honest, 1-sided- and 2-sided VSE. 
The results are shown below:

===============      ============
Election Method       Average VSE
===============      ============             
plurality                0.515
top_two                  0.683
irv                      0.725
score                    0.762
approval50               0.772
maj_judge                0.812
ranked_pairs             0.849
smith_minimax            0.850
star5                    0.858
smith_score              0.870
===============      ============

Results show that Condorcet systems such as ranked_pairs and smith_minimax, and smith_score are excellent performers.
STAR voting is also a top performing system. 
The worst performing systems are plurality, top-two, and instant-runoff (IRV). 


