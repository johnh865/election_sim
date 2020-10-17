# -*- coding: utf-8 -*-
"""
Repository of voting methods. All voting methods here have the same
function interface and are interchangeable. 



Attributes
-----------
ranked_methods : dict
    Collection of ranked voting methods with their name as the key and 
    the voting system's function as the value.
scored_methods : dict
    Collection of ranked voting methods with their name as the key and 
    the voting system's function as the value.
vote_methods : dict
    Collection of single-mark voting methods with their name as the key and 
    the voting system's function as the value.    
all_methods : dict
    Collection of all available voting methods with their name as the key and 
    the voting system's function as the value.    
            
"""

# from . import irv
# from . import plurality
# from . import score
# from . import tools
# from . import condorcet
# from . import condcalcs


# from .voterunner import eRunner
# from .voterunner import (ranked_methods, scored_methods,
#                          vote_methods, all_methods)

from votesim.votemethods.voterunner import eRunner
from votesim.votemethods.methodinfo import (ranked_methods, 
                                            rated_methods,
                                            scored_methods,
                                            vote_methods, 
                                            all_methods,    
                                            method_keywords,
                                            get_ballot_type,
                                            )
from votesim.votemethods import (irv,
                                 plurality,
                                 score,
                                 tools,
                                 condorcet,
                                 condcalcs,
                                 )
