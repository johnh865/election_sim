.. _glossary:

        

Glossary
========

.. glossary::

   benchmark
       A suite of multiple election simulations for a specific voter model,
       in which particular parameters are varied.  
       
   voting system
       An voting method such as plurality, instant-runoff-voting, 
       scored voting, or otherwise. 

   plurality ratio
	  The ratio of a plurality winner's honest votes 
	  over the total votes cast in the election. 
	  
   plurality winner ratio
      See plurality ratio.
      
   Condorcet winner
       The candidate which conforms to the
       `Condorcet Criterion <https://en.wikipedia.org/wiki/Condorcet_criterion>`_
       
   majority winner
       The candidate who recieves greater than 50% of 1st choice votes. 
       
   plurality winner
       The candidate who recieves the most honest 1st choice votes.
   
   utility winner
       The candidate that maximizes the average utiltiy of the voter population.
   
   C scenario
       An election scenario in which the Condorcet winner is not the utility
       winner. 
   
   CPU scenario
       An election scenario in which a Condorcet-plurality-utility winner 
       is found; where the the Condorcet/plurality/utility winner are 
       coincidental. 

   CU scenario 
       An election scenario in which the
       utility winner is also the Condorcet winner. 
       
   CP scenario
       An election scenario in which the plurality winner is also the 
       Condorcet winner. However, the Condorcet-plurality winner is not
       the utility winner. 
       
   MU scenario
       An election scenario in which a majority winner exists and is also
       the utility winner.
       
   M scenario
       An election scenario in which a majority winner exists but is not
       the utility winner. 
   
   nc scenario
       An election scenario with a 
       `Condorcet cycle. <https://en.wikipedia.org/wiki/Condorcet_paradox>`_
       In other words No Condorcet winner is found. 
       
       
   Voter Satisfaction Efficiency (VSE)
       Measure of voter average utility in which at 100% VSE, the candidate
       that maximizes utility is elected. At less than 0% VSE, a candidate 
       with utility worse than the average candidate is selected. This
       metric was proposed by 
       `Jameson Quinn <https://electionscience.github.io/vse-sim/VSE/>`_.
       
   Voter Regret (VR)
       Measure of voter utility proposed by John Huang. 
       At 0% voter regret, the candidate that maximizes voter utility 
       is elected. When VR is greater than 100%, a candidate is elected 
       whose regret exceeds the difference between electing a candidate 
       with preferences of a random, average voter and an ideal candidate
       located in the median centroid of the voter population. 
       
       
   voter tolerance
       The maximum difference in utility between the voter and the candidate
       in which the voter will reward the candidate with a rating greater 
       than zero. At utility differences greater than the voter tolerance,
       candidates recieve zero score or recieve no ranking. 
       
       

       
