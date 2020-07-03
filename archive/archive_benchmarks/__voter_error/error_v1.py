
import os
import itertools
import pandas as pd
import numpy as np

import votesim
from votesim.models import spatial
from votesim.benchmarks.runtools import benchrun


BENCHMARK_NAME = 'error_v1'
OUTPUT_FILE = BENCHMARK_NAME + '-%s.pkl.gz'


def model(methods, 
          ndim,
          error_mean,
          error_width,
          clim_mean,
          clim_width,
          seed=1,
          cnum=8,
          trialnum=1000,
          numvoters=100,
          strategy='candidate', 
          
          ):
    e = spatial.Election(None, None, seed=seed, name=BENCHMARK_NAME)
    v = spatial.ErrorVoters(seed=seed, strategy=strategy)
    v.add_random(numvoters=numvoters, 
                 ndim=ndim,
                 error_mean=error_mean,
                 error_width=error_width,
                 clim_mean=clim_mean,
                 clim_width=clim_width,
                 )
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial)
        c.add_random(cnum, sdev=3)
        e.set_models(voters=v, candidates=c)
        e.user_data(num_dimensions=ndim,
                    num_candidates=cnum,
                    error_mean=error_mean,
                    error_width=error_width,
                    )
        
        for method in methods:
            e.run(etype=method)
    return e
        

def case_args(methods):
    seed = 1
    numvoters=100
    trialnum=1000
    ndims = np.arange(1, 4)
    cnum = 8
    clim_mean = -1
    clim_width = 0
    error_means = [0, .05, .1, .2, .3]
    error_widths = [0, .05, .1, .2, .3]
    
    iters = itertools.product(ndims, error_means, error_widths)
    for x in iters:
        ndim = x[0]
        emean = x[1]
        ewidth = x[2]
        args = (methods, ndim, emean, ewidth, 
                clim_mean, clim_width,
                seed, cnum, trialnum, numvoters)
        yield args
        
    
def run(methods, cpus=1, filename=OUTPUT_FILE):
    
    return benchrun(methods, 
                     model=model,
                     case_args=case_args,
                     cpus=cpus,
                     filename=filename)        
    

def post(filename='simpleNd-*'):
    
    ###############################################################################
    
    
    percentiles = np.arange(100)
    categories = [
             'args.candidate.1.add_random.cnum',
             'args.voter.1.add_random.ndim',
             'args.voter.0.init.strategy',
             'args.election.1.run.etype'
            ]    
    
    metric_name = 'stats.regret.efficiency_voter'

    ###############################################################################
    ### Open saved data files
    
    print('Opening saved election data')
    d1 = os.getcwd()
    filenames = votesim.utilities.misc.detectfiles(d1, filename)
    for fn in filenames:
        print(os.path.basename(fn))
    
    
    
    dframes = {}
    for fname in filenames:
        dfi = pd.read_pickle(fname,)
        etype = dfi['args.election.1.run.etype'].iloc[0]
        dframes[etype] = dfi
        
    df = pd.concat(dframes)
    df = df.infer_objects()
    
    ###############################################################################'
    ### Get Satisfaction percentiles
    
    output = {}
    output['percentiles'] = percentiles
    for method, dfi in dframes.items():
        vse = dfi[metric_name]
        p = np.percentile(vse, percentiles)
        output[method] = p
    
    fname = 'simpleNd_vse_percentiles.csv'
    header = 'Voter satisfaction percentiles normalized by voter populaiton'
    df2 = pd.DataFrame(data=output)
    df2.to_csv(fname, header=header)
                
    ###############################################################################'
    ### Get average satisfaction by simulation arguments
    
    print('Average satisfaction for various parameters')
    dfp = pd.pivot_table(df, 
                         index=categories,
                         values=metric_name,
                         aggfunc=np.mean,
                         )
    
    fname = 'simpleNd_vse_categories.csv'
    header = 'Voter satisfaction averages categorized by simulator parameters'
    dfp.to_csv(fname, header=header)
    return

    


def cli():
    """Command line user interface"""
    methods = votesim.votesystems.all_methods.keys()
    methods = list(methods)
    
    print('Simple N-dimensional voting methods benchmark')
    print('Available voting methods:')
    for ii, m in enumerate(methods):
        print('%5d.' % ii, m)
    
    print('\nWhich voting methods would you like to assess?')
    print('Enter desired methods By their number, in a comma delimited list')
    s = input('Desired methods: ')
    
    a = s.split(',')
    a = [ai for ai in a if ai != '']
    a = np.array(a, dtype=int)
    
    methods1 = []
    for ai in a:
        try:
            methods1.append(methods[ai])
        except IndexError:
            raise Exception('Method Number not in list')
    run(methods1)
    
  
if __name__ == '__main__':
    cli()
    post()
    pass




