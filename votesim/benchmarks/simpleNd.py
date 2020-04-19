"""
"""

import os
import itertools
import pandas as pd
import numpy as np

import votesim
from votesim.models import spatial
from votesim.utilities import lazy_property
import votesim.benchmarks.runtools as runtools

# BENCHMARK_NAME = 'simpleNd'
# OUTPUT_FILE = BENCHMARK_NAME + '-%s.pkl.gz'



def model(name, seed, numvoters, trialnum, ndim, strategy, cnum, methods):
    """Define election model here"""

    e = spatial.Election(None, None, seed=seed, name=name)
    v = spatial.SimpleVoters(seed=seed, strategy=strategy)
    v.add_random(numvoters, ndim=ndim)
    
    for trial in range(trialnum):
        c = spatial.Candidates(v, seed=trial)
        c.add_random(cnum, sdev=1.5)
        c.add_random(cnum, sdev=3)
        e.set_models(voters=v, candidates=c)
        
        # Save parameters
        e.user_data(
                    num_voters=numvoters,
                    num_candidates2x=cnum,
                    num_dimensions=ndim,
                    strategy=strategy,
                    )
        
        for method in methods:
            e.run(etype=method)
            
    return e

        
        
# class CaseGenerator(object):
#     """
#     Flexible case generator that allows you to specify different parameters
    
#     """
#     def __init__(self,
#                  seed=0,
#                  numvoters=100,
#                  trialnum=1000,
#                  ndims=(1, 2, 3, 4, 5),
#                  strategies=('candidate','voter'),
#                  cnums=(2, 3, 4, 5, 6, 7, ),
#                  ):
#         self.seed = seed
#         self.numvoters = numvoters
#         self.trialnum = trialnum
#         self.ndims = ndims
#         self.strategies  = strategies
#         self.cnums = cnums
#         self._iters = itertools.product(strategies, ndims, cnums)    
#         return
    
    
#     def __call__(self, methods):
#         iters = self._iters
#         for x in iters:
#             strategy = x[0]
#             ndim = x[1]
#             cnum = x[2]
#             args = (self.seed, self.numvoters, self.trialnum, 
#                     ndim, strategy, cnum, methods)
#             yield args           
            
  


# def run2(methods,
#          cpus=1, 
#          filename=OUTPUT_FILE, 
#          seed=0,
#          numvoters=100, 
#          trialnum=1000, 
#          ndims=(1, 2, 3, 4, 5),
#          strategies = ('candidate', 'voter'),
#          cnums=tuple(np.arange(2, 8)),
#          ):
#     """Run model with a bit more parametric flexibility"""
#     case_args = CaseGenerator(
#                               seed=seed,
#                               numvoters=numvoters, 
#                               trialnum=trialnum,
#                               ndims=ndims,
#                               strategies=strategies,
#                               cnums=cnums,
#                               )
#     return runtools.benchrun(methods, 
#                      model=model,
#                      case_args=case_args,
#                      cpus=cpus,
#                      filename=filename)


class CaseGenerator(object):
    """Generate arguments for input into model.
    
    Parameters
    ----------
    **kwargs : dict
        dict of parameters. If the value is iterable, it will be
        iterated upon using `itertools.product`
    
    """
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            is_iter = True
            
            if isinstance(v, str):
                is_iter = False
            try:
                iter(v)
            except TypeError:
                is_iter = False
            
            if not is_iter:
                kwargs[k] = [v]
                
        self.kwargs = kwargs
        
        # iters = itertools.product(*self.kwargs.values())
        # self._iters = iters
        self._keys = kwargs.keys()
        return            
    
    
    def __call__(self, methods):
        """Build a iterator that generates arguments for the benchmark model"""
        iters =  itertools.product(*self.kwargs.values())
        for xargs in iters:
            d = dict(zip(self._keys, xargs))
            d['methods'] = methods
            yield d



# class CaseGenerator(object):
#     def __init__(self, kwargs):
#         for k, v in kwargs.items():
#             is_iter = True
            
#             if isinstance(v, str):
#                 is_iter = False
#             try:
#                 iter(v)
#             except TypeError:
#                 is_iter = False
            
#             if not is_iter:
#                 kwargs[k] = [v]    
#         self.kwargs = kwargs
#         return
    
#     def _build_generator(self):
#         iters = itertools.produ
    


class SimpleCreate(object):
    """Base object for creating a benchmark"""
    def __init__(self, name, model, case_args):
        self.name = name
        self.model = model
        self.case_args = case_args
        
        self.output_file = name + '-run-%s.pkl.gz'
        self.output_pattern = self.output_file.replace('%s', '*')
        
        self.post_file = name + '-post-categories.pkl.gz'        
        self.plot_file = name + '-plot-category-%d.png'        
        
        #self._case_args = CaseGenerator(**kwargs)
        return
    
    def run(self, methods, cpus=1, filename=''):
        if filename == '':
            filename = self.output_file
        
        return runtools.benchrun(methods,
                                 model=self.model, 
                                 case_args = self.case_args,
                                 cpus=cpus, 
                                 filename=filename)    
    
    
    def post(self, pattern='', post_file=''):
        """Post process benchmark
        
        Parameters
        ----------
        pattern : str
            File pattern to detect
        post_file : str
            Post-processed dataframe file to create
        """
        if pattern == '':
            pattern = self.output_pattern
        if post_file == '':
            post_file = self.post_file
        
        p = runtools.PostProcessor(pattern,)
        p.parameter_stats(post_file)    
        self.post_file = post_file
        return p
    
    
    def plot(self, 
             post_file='',
             plot_file='', 
             x_axis='num_candidates',
             y_axis='etype',
             key='output.regret.efficiency_voter',
             func='subtract100',
             **kwargs):
        
        if post_file=='':
            post_file = self.post_file
        if plot_file == '':
            plot_file = self.plot_file
            
        df1 = pd.read_pickle(post_file)
        f = runtools.heatplots(df1, 
                           filename=plot_file,
                           x_axis=x_axis,
                           y_axis=y_axis,
                           key=key,
                           func=func,
                           **kwargs
                           )
        return f
    



class Simple1:
    """SimpleNd benchmark #1
    
    Features
    ---------
    
    - Spatial model
    - Normal voter preference distribution
    - 1-5 spatial preference dimensions
    - 2-8 candidates within 1.5 std deviations of voter preferences
    - 2-8 candidates within 3.0 std deviations of voter preferences
    - 2 voter strategies.
    
     #1, 1-5 dimensions, (2-8) candidates x 2'
    """
    name = 'simple-1'
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = 0
    kwargs['numvoters'] = 100
    kwargs['trialnum'] = 1000
    kwargs['ndim'] = np.arange(1, 6)
    kwargs['strategy'] = ('candidate', 'voter')
    kwargs['cnum'] = np.arange(2, 8)
    case_args = CaseGenerator(**kwargs)
    _Simple1 = SimpleCreate(name, model, case_args)
    
    run = _Simple1.run
    post = _Simple1.post
    plot = _Simple1.plot



class SimplePop:
    name = 'simple-pop'
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = np.arange(0, 20)
    kwargs['numvoters'] = [10, 25, 450, 100, 200, 400]
    kwargs['trialnum'] = [300]
    kwargs['ndim'] = 2
    kwargs['strategy'] = 'candidate'
    kwargs['cnum'] = 6
    
    case_args = CaseGenerator(**kwargs)
    _Simple1 = SimpleCreate(name, model, case_args)
    run = _Simple1.run
    post = _Simple1.post
    plot = _Simple1.plot
    
    
    
class SimpleDummy:
    """
    Dummy benchmark for testing purposes. Runs a tiny number of cases.
    """
    name = 'simple-dummy'
    kwargs = {}
    kwargs['name'] = name
    kwargs['seed'] = 0
    kwargs['numvoters'] = 10
    kwargs['trialnum'] = 10
    kwargs['ndim'] = (1, 2)
    kwargs['strategy'] = ('candidate', 'voter')
    kwargs['cnum'] = (2, 3, 4)
    case_args = CaseGenerator(**kwargs)
    _Simple1 = SimpleCreate(name, model, case_args)
    
    run = _Simple1.run
    post = _Simple1.post
    plot = _Simple1.plot






class _SimpleDummy_TEST:
    """Container for dummy simple benchmark, used for testing"""
    
    benchmark_name = 'simple_dummy'
    output_file = benchmark_name + '-%s.pkl.gz'
    output_pattern = output_file.replace('%s', '*')
    
    post_file = output_file % 'post-categories'
    plot_file = benchmark_name + '-plot-category-%d.png'

    
    @staticmethod
    def run(methods, cpus=1, filename=output_file):
        return run2(methods, 
                    cpus = cpus,
                    filename = filename,
                    seed = 0,
                    numvoters = 10,
                    trialnum = 10,
                    ndims = (1,2),
                    cnums = (2,3,4),
                    )
    
    
    def post(pattern=output_pattern, 
             post_file=post_file):
        p = runtools.PostProcessor(pattern,)
        p.parameter_stats(post_file)
        
        pass









def _post_OLD(filename='simpleNd-*'):
    
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
    print('saving file %s' % fname)
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
    print('saving file %s' % fname)
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
    print('Enter desired methods By their number, in a comma delimited string')
    s = input('Desired methods: ')
    
    a = s.split(',')
    a = [ai for ai in a if ai != '']
    a = np.array(a, dtype=int)
    
    cpus = input('Number of processes: ')
    cpus = int(cpus)
    methods1 = []
    for ai in a:
        try:
            methods1.append(methods[ai])
        except IndexError:
            raise Exception('Method Number not in list')
    run(methods1, cpus=cpus)
    
  
if __name__ == '__main__':
# #    run(['plurality'], cpus=4)
#     cli()
#     post()
    pass




