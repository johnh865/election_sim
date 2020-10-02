"""
Compare elections.
"""
from votesim.models import spatial
from votesim.metrics import ElectionStats, BaseStats
from votesim.models.spatial import Election


class TacticStats(BaseStats):
    def __init__(self, stats1: ElectionStats, stats2: ElectionStats):
        ed1 = stats1.electionData
        ed2 = stats2.electionData
        self._distances1 = ed1.distances
        self._distances2 = ed2.distances
        self._stats1 = stats1
        self._stats2 = stats2
        
        
    def regret_change(self):
        winner = self._stats1.winner
        dist_change = self._distances2 - self._distances1
        return dist_change[: winner]
        
    
        
        
        
        