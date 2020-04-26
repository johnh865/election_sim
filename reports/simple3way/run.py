import pandas as pd
import votesim
import os
from votesim.benchmarks.simpleNd import SimpleThreeWay

benchmark = SimpleThreeWay

methods = votesim.votesystems.all_methods
methods = [
    
    'smith_minimax',
    'ranked_pairs',
    'irv',
    # 'irv_stv',
    'top_two',
    # 'rrv',
    # 'sequential_monroe',
    # 'score',
    # 'star',
    # 'majority_judgment',
    'smith_score',
    'approval100',
    'approval75',
    'approval50',
    'score5',
    'score10',
    'star5',
    # 'star10',
    'plurality',
    
    ]
dirname = votesim.definitions.DIR_DATA_BENCHMARKS
dirname = os.path.join(dirname, benchmark.name)


def run():    


    # benchmark.run(methods, cpus=4,)

    os.makedirs(dirname, exist_ok=True)
    benchmark.run(methods, cpus=8, dirname=dirname)
    p = benchmark.post()
    return p
    
    
def read():
    return benchmark.read(dirname=dirname)


def plot():
    benchmark.plot()
    
    
keys =  [
 'args.election.name',
 'args.election.etype',
 'args.user.num_voters',
 'args.user.num_candidates',
 'args.user.num_dimensions',
 'args.user.strategy',
 'args.user.voter_tolerance',
 'output.voter.mean',
 'output.voter.median',
 'output.voter.regret_mean',
 'output.voter.regret_median',
 'output.voter.regret_random_avg',
 'output.voter.regret_std',
 'output.voter.std',
 'output.candidate.best',
 'output.candidate.preference',
 'output.candidate.regret_random',
 'output.candidate.regrets',
 'output.regret.PR',
 'output.regret.PR_std',
 'output.regret.best',
 'output.regret.consensus',
 'output.regret.efficiency_candidate',
 'output.regret.efficiency_voter',
 'output.regret.normed',
 'output.winner.all',
 'output.winner.num',
 'output.ballot.bullet_num',
 'output.ballot.bullet_ratio',
 'output.ballot.full_num',
 'output.ballot.full_ratio',
 'output.ballot.marked_avg',
 'output.ballot.marked_num',
 'output.ballot.marked_std',
 'output.ties']

    
if __name__ == '__main__':
    p = read()
    
    df = pd.DataFrame()
    df = p.dataframe
    gb = df.groupby('args.election.etype')
    etypes = list(gb.groups.keys())
    parameters = p.user_parameters
    df0 = gb.get_group(etypes[0])
    
    