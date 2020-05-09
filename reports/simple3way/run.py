import pandas as pd
import votesim
import os
from votesim.benchmarks.simpleNd import SimpleThreeWay
from  votesim.benchmarks import runtools
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
 'args.name',
 'args.etype',
 'args.user.num_voters',
 'args.user.num_candidates',
 'args.user.num_dimensions',
 'args.user.strategy',
 'args.user.voter_tolerance',
 'output.voter.data_dependencies',
 'output.voter.pref_mean',
 'output.voter.pref_median',
 'output.voter.pref_std',
 'output.voter.regret_mean',
 'output.voter.regret_median',
 'output.voter.regret_random_avg',
 'output.candidate.data_dependencies',
 'output.candidate.pref',
 'output.candidate.regret_avg',
 'output.candidate.regret_best',
 'output.candidate.regrets',
 'output.candidate.winner_condorcet',
 'output.candidate.winner_majority',
 'output.candidate.winner_plurality',
 'output.candidate.winner_utility',
 'output.winner.data_dependencies',
 'output.winner.regret',
 'output.winner.regret_efficiency_candidate',
 'output.winner.regret_efficiency_voter',
 'output.winner.regret_normed',
 'output.winner.ties',
 'output.winner.winners',
 'output.winner_categories.data_dependencies',
 'output.winner_categories.is_condorcet',
 'output.winner_categories.is_majority',
 'output.winner_categories.is_utility',
 'output.ballot.bullet_num',
 'output.ballot.bullet_ratio',
 'output.ballot.data_dependencies',
 'output.ballot.full_num',
 'output.ballot.full_ratio',
 'output.ballot.marked_avg',
 'output.ballot.marked_num',
 'output.ballot.marked_std'
 ]

    
if __name__ == '__main__':
    run()
    # p = read()
    
    # df = pd.DataFrame()
    # df = p.dataframe
    
    # # ii_maj = df['output.candidate.winner_majority'] < 0 
    # # ii_maj = df['output.candidate.winner_majority'] > -1 
    # # df2 = df.loc[ii_maj]    
    # df2 = df

    
    # post = runtools.PostProcessor(df=df2)
    # df3 = post.parameter_stats()
        
    # # etypes = list(gb.groups.keys())
    # # parameters = p.user_parameters

    # runtools.heatplots(df3, 
    #                    'plot-output.png', 
    #                    x_axis='voter_tolerance',
    #                    y_axis='etype',
    #                    key='output.winner.regret_efficiency_voter',
    #                    ncols=1,
    #                    func='subtract100'
    #                    )
    
    
    
    
    
    