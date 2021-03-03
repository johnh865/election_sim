# -*- coding: utf-8 -*-
import numpy as np
from votesim.votemethods.ranked import borda
    
    
def test_wiki_1():
    """Test wiki example,
    https://en.wikipedia.org/wiki/Borda_count, Mar-3-2021
    """
    # Andrew Brian Catherine David
    # A B C D
    d = (
        [[1, 3, 2, 4]] * 51 +
        [[4, 2, 1, 3]] * 5 +
        [[4, 1, 2, 3]] * 23 +
        [[4, 3, 2, 1]] * 21
        )
    
    d = np.array(d)
    winners, ties, output = borda(d)
    
    assert len(winners) == 1
    assert winners[0] == 2
    
    correct_tally = np.array([153, 151, 205, 91])
    assert np.all(output['tally'] == correct_tally)
    return



def test_wiki_2():
    """Test wiki example #2.

    https://en.wikipedia.org/wiki/Borda_count, Mar-3-2021
    """
    
    # M N C K
    d = (
        [[1, 2, 3, 4]]*42 + 
        [[4, 1, 2, 3]]*26 + 
        [[4, 3, 1, 2]]*15 + 
        [[4, 3, 2, 1]]*17
        )
    d = np.array(d)
    winners, ties, output = borda(d)
    
    assert len(winners) == 1
    assert winners[0] == 1
    correct_tally = np.array([126, 194, 173, 107])
    assert np.all(output['tally'] == correct_tally)

    


if __name__ == '__main__':
    test_wiki_1()
    test_wiki_2()