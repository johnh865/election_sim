        
        
class ___OLD_Election(object):
    """
    Simulate elections. 
    
    1. Create voters
    -----------------
    Voters can be created with methods:
        
    - set_random_voters
    - set_voters
    
    2. Create candidates
    --------------------
    Candidates can be randomly created or prescribed using methods:
    
    - generate_candidates
    - add_candidates
    - add_median_candidate
    - add_faction_candidate
    
    3. Run the election
    ---------------------
    Use `run` to run an election method of your choice. All historical output
    is recorded in `self._result_history`.
    
    
    4. Retrieve election output
    ----------------------------
    Output is stored in the attributes:
        
    - self.result -- dict of dict of various metrics and arguments. 
    - self.dataseries() -- construct a Pandas Series from self.result
    - self.results() -- Pandas DataFrame of current and previously run results.  
    
    """
    def __init__(self, seeds=(None,None,None)):
        
        self._args = {}
        self.set_seed(*seeds)
        self._stats = metrics.ElectionStats()
        self._result_history = []
        
        return
    
    
    def set_user_args(self, kwargs):
        """Add user defined dict of arguments to database"""
        self._args = kwargs
    
    
    def set_seed(self, voters, candidates=None, election=None):
        """
        Set random seed for voter generation, candidate generation, and running elections. 
        """
        if candidates is None:
            candidates = voters
        if election is None:
            election = voters
            
        self._seed_voters = voters
        self._seed_candidates = candidates
        self._seed_election = election
        return
    
    

    @staticmethod
    def _RandomState(seed, level):
        """
        Create random state.
        Generate multiple random statse from a single seed, by specifying
        different levels for different parts of Election. 
        
        Parameters
        ----------
        seed : int
            Integer seed
        level : int
            Anoter integer seed.
        """
        if seed is None:
            return np.random.RandomState()
        else:
            return np.random.RandomState((seed, level))
        
    
    
    
    def set_random_voters(self, ndim, nfactions,
                        size_mean=100, 
                        size_std=1.0, 
                        width_mean=1.0,
                        width_std=0.5,
                        tol_mean = 1.0,
                        tol_std = 0.5,
                        error_std = 0.0,):
        """
        Parameters
        -----------
        ndim : int
            Number of preference dimensions
        nfactions : int
            Number of voter factions
        size_mean : int
            Average number of voters per faction
        size_std : float
            Std deviation of number of voters per faction. 
        width_mean : float
            Average preference width/scale of faction normal distribution.
        width_std : float
            Std deivation of preference width/scale of faction normal distribution.
        seed : None or int
            Random state seed
        """
        seed = self._seed_voters
        rs = self._RandomState(seed, 1)

        # generation faction centroid coordinates
        coords = rs.uniform(-1, 1, size=(nfactions, ndim))
        sizes = ltruncnorm(
                           loc=size_mean,
                           scale=size_std * size_mean, 
                           size=nfactions,
                           random_state=rs
                           )
        sizes = sizes.astype(int)
        sizes = np.maximum(1, sizes)  # Make sure at least one voter in each faction
        
        widths = ltruncnorm(
                            loc=width_mean,
                            scale=width_std,
                            size=nfactions * ndim
                            )
        
        widths = np.reshape(widths, (nfactions, ndim))
        
        logger.debug('coords=\n %s', coords)
        logger.debug('sizes=\n %s', sizes)
        logger.debug('widths=\n %s', widths)

        self.set_voters(coords, 
                        sizes, 
                        widths,
                        tol_mean=tol_mean,
                        tol_std=tol_std, 
                        error_std=error_std,
                        )
        return 
    

    
    def set_voters(self, coords, sizes, widths,
                   tol_mean=1., tol_std=1., error_std=0.,):
        """
        Parameters
        ----------
        coords : array shaped (a, b) 
            Centroids of a faction voter preferences.
        
            - rows `a` = coordinate for each faction
            - columns `b' = preference dimensions. The more columns, the more preference dimensions. 
            
        sizes : array shaped (a,)
            Number of voters within each faction, with a total of `a` factions.
            Use this array to specify how many people are in each faction.
        widths : array shaped (a, b)
            The preference spread, width, or scale of the faction. These spreads
            may be multidimensional. Use columns to specify additional dimensions.         
        
        """
        seed = self._seed_voters
        rs = self._RandomState(seed, 2)
        
        coords = np.array(coords)
        sizes = np.array(sizes)
        widths = np.array(widths)
        
        voters = gaussian_preferences(coords, sizes, widths, rstate=rs)
#        tolerance = rs.normal(tol_mean, tol_std, size=voters.shape[0])
#        error = np.abs(rs.normal(0, error_std, size=voters.shape[0]))
        
        tolerance = ltruncnorm(loc=tol_mean,
                               scale=tol_std,
                               size=voters.shape[0],
                               random_state=rs)
        error = ltruncnorm(loc=tol_mean,
                            scale=tol_std,
                            size=voters.shape[0],
                            random_state=rs)
        
        self._stats.run(voters)
        
        self.voters = voters
        self.tolerance = tolerance
        self.error = error
        
        voter_args = {}
        voter_args['coords'] = coords
        voter_args['sizes'] = sizes
        voter_args['widths'] = widths
        voter_args['tol_mean'] = tol_mean
        voter_args['tol_std'] = tol_std
        voter_args['error_std'] = error_std
        voter_args['seed'] = seed
        voter_args['ndim'] = coords.shape[1]
        
        self._voter_args = voter_args      
        return 
    
    


    
    def run(self, etype=None, method=None, btype=None, 
            numwinners=1, scoremax=None, kwargs=None):
        
        """Run the election & obtain results. For ties, randomly choose winner. 
        
        Parameters
        ----------
        etype : str
            Name of election type.
            Mutually exclusive with `method` and `btype`
            Supports the following election types, for example:
                
            - 'rrv' -- Reweighted range and pure score voting
            - 'irv' -- Single tranferable vote, instant runoff.
            - 'plurality' -- Traditional plurality & Single No Transferable Vote.
            
                
        method : func
            Voting method function. Takes in argument `data` array shaped (a, b)
            for (voters, candidates) as well as additional kwargs.
            Mutually exclusive with `etype`.
            
            >>> out = method(data, numwin=self.numwinneres, **kwargs)
            
        btype : str
            Voting method's ballot type. 
            Mutually exclusive with `etype`, use with `method`.
            
            - 'rank' -- Use candidate ranking from 1 (first place) to n (last plast), with 0 for unranked.
            - 'score' -- Use candidate rating/scored method.
            - 'vote' -- Use traditional, single-selection vote. Vote for one (1), everyone else zero (0).  
        """
        seed = self._seed_election
        
        run_args = {}
        run_args['etype'] = etype
        run_args['method'] = method
        run_args['btype'] = btype
        run_args['numwinners'] = numwinners
        run_args['scoremax'] = scoremax
        run_args['kwargs'] = kwargs
        run_args['seed'] = seed
        
        if seed is None:
            seed2 = seed
        else:
            seed2 = (seed, 4)
        
        
        e = ElectionRun(self.voters, 
                        self.candidates,
                        numwinners=numwinners,
                        cnum=None,
                        error=self.error,
                        tol=self.tolerance)    
        
        e.run(etype, 
              method=method,
              btype=btype, 
              scoremax=scoremax, 
              seed=seed2,
              kwargs=kwargs)
        
        stats = metrics.ElectionStats(voters=self.voters,
                                      candidates=self.candidates,
                                      winners=e.winners,
                                      ballots=e.ballots)        
        
        ### Build dictionary of all arguments and results 
        results = {}
        results['args.candidate'] = self._candidate_args
        results['args.voter'] = self._voter_args
        results['args.election'] = run_args
        
        for key, val in self._args.items():
            newkey = 'args.user.' + key
            results[newkey] = val
        
        results['stats'] = stats.stats
        results['stats']['ties'] = e.ties
        results = utilities.flatten_dict(results, sep='.')
        self.results = results        
        
        self._result_history.append(results)
        return results
    

    def dataseries(self):
        """Retrieve pandas data series of output data"""        
        return pd.Series(self.results)
    
    
    def dataframe(self):
        """Construct data frame from results history"""
        
        series = []
        for r in self._result_history:
            series.append(pd.Series(r))
        df = pd.concat(series, axis=1).transpose()
        self._dataframe = df
        return df
    
    def save(self, name):
        self._dataframe.to_json(name)
        
        
    def rerun(**kwargs):
        """Re-run election using dataframe output"""
        
        d = kwargs.copy()
        for k in d:
            if not k.startswith('args.'):
                d.pop(k)
        
        
        e = Election()
        self.candidates = d['args.candidate.coords']
        

        
        
        
        
        
    
#def build_dataframe(results):
#    """Build a dataframe from a list of Election.results
#    
#    Parameters
#    -----------
#    elections : list of Election.results
#        After election has been run
#        
#    
#    """
#
#
#    a = [e.dataseries() for e in elections]
#    df = pd.concat(a, axis=1).transpose()        
#    return df
#
#        
#    
#
#
 
class __Voters(object):
    """
    Create voters with preferneces. Each voter has the following properties:
    
    Voter Properties
    ----------------
    - Preference coordinates, n-dimensional
        The voter's preference location in spatial space. 
        
    - Preference tolerance, n-dimensional
        How tolerant the voter is to other preferences in spatial space. 
        
    - Preference weight, n-dimensional
        How much the voter cares about a particular issue.
        
    - Error
        The likely amount of error the voter will make when estimating
        preference distance from themselves to a candidate. 
        
    - Candidate Limit
        The max number of candidates a voters is willing to consider - on the 
        assumption that every voter has limited mental & research resources.
        
        
    Voter Distribution Properties
    ------------------------------
    Distributions of voters shall be created using multiple
    normal distributions. 
    
    For each voter property, there may be
    
    - mean -- The mean value of a property's distribution centroid. 
    - std -- The standard deviation of a distribution's centroid.
    - width -- The mean value of a distribution's dispersion or width
    - width_std -- The standard deviation of a distribution's dispersion or width.
    
    
    Attributes
    ---------
    voters : array shape (a, b)
        Voter preference for `a` voter num & `b` dimensions num. 
    tolerance : array shape (a, b)
        Voter tolerance for `a` voter num  and `b` dimensions num.
    error : array shape (a,)
        Voter error for `a` voters.
    weight : array shape (a, b)
        Voter dimension weights for `a` voters and `b` dimensions. 
        
    
    """
    def __init__(self, seed=None):
        self._method_records = utilities.RecordActionCache()
        self.set_seed(seed)
        

    @utilities.recorder.record_actions()
    def set_seed(self, seed):
        """ Set pseudorandom seed """
        self.seed = seed
        self._randomstate = _RandomState(seed, VOTERS_BASE_SEED)  
        self._randomstate2 = _RandomState(seed, CLIMIT_BASE_SEED)  
        return
        
    
    @utilities.recorder.record_actions()
    def add_faction(self,
                    coord,
                    size, 
                    width, 
                    tol_mean,
                    tol_width, 
                    error_width=0.0,
                    weight_mean=1.0,
                    weight_width=0.0):
        """Add a faction of normally distributed voters
        
        Parameters
        ----------
        coord : array shape (a,)
            Faction centroid preference coordinates
 
        sizes : int
            Number of voters within each faction, 
            
        width : float or array shape (a,)
            The preference spread, width, or scale of the faction. These spreads
            may be multidimensional. Use columns to specify additional dimensions.    
            
        
            
        """
   
        p, t, e, w, c = self._create_distribution(
                                               coord, size, width, 
                                               tol_mean, 
                                               tol_width,
                                               error_width,
                                               weight_mean,
                                               weight_width
                                               )
        
        try:
            self.voters = np.row_stack((self.voters, p))
            self.tolerance = np.row_stack((self.tolerance, t))
            self.error = np.append(self.error, e)
            self.weight = np.row_stack((self.weight, w))
            self.fcoords = np.row_stack((self.fcoords, coord))
            self.climit = np.append(self.climit, c)
            
        except AttributeError:
            self.voters = np.atleast_2d(p)
            self.tolerance = t
            self.error = e
            self.weight = np.atleast_2d(w)
            self.fcoords = np.atleast_2d(coord)
            self.climit = c
            
            
        return
    
    
    
    
    def _create_distribution(self,
                             coord, size, width, 
                             tol_mean, 
                             tol_std, 
                             error_std=0.0,
                             weight_mean=1.0, 
                             weight_std=1.0,
                             cnum_mean=np.nan,
                             cnum_std=1.0
                             ):
        """Perform calculations for add_faction"""
        
        rs = self._randomstate
        coord = np.array(coord)
        ndim = len(coord)
        
        preferences = rs.normal(
                               loc=coord,
                               scale=width,
                               size=(size, ndim),
                               )
        tolerance = ltruncnorm(
                               loc=tol_mean,
                               scale=tol_std,
                               size=size,
                               random_state=rs,
                               )
        error = rs.normal(
                           loc=0.0,
                           scale=error_std,
                           size=size,
                           )
        
        weight = ltruncnorm(
                            loc=weight_mean,
                            scale=weight_std,
                            size=(size, ndim),
                            random_state=rs,
                            )

        climit = ltruncnorm(loc=cnum_mean,
                             scale=cnum_std,
                             size=size,
                             random_state=rs)
        out = (preferences,
               tolerance,
               error,
               weight,
               climit)
        return out
        
    
    def calc_ratings(self, candidates):
        """
        Calculate preference distances & candidate ratings for a given set of candidates
        """
        try:
            candidates = candidates.candidates
        except AttributeError:
            pass
        
        voters = self.voters
        weights = self.weight
        error = self.error
        tol = self.tolerance
        
        if voters.shape[1] == 1:
            weights = None
        rstate = self._randomstate
        
        distances = vcalcs.voter_distances(voters, 
                                           candidates,
                                           weights=weights)
        
        distances = vcalcs.voter_distance_error(distances, 
                                                error,
                                                rstate=rstate)
        
        ratings = vcalcs.voter_scores_by_tolerance(
                                                   None, None,
                                                   distances=distances,
                                                   tol=tol,
                                                   cnum=None,
                                                   strategy='abs',
                                                   )
        self.ratings = ratings
        self.distances = distances
        return ratings
    
    
    def stats(self):
        s = metrics.ElectionStats(voters=self.voters,
                                  weights=self.weights
                                  )
        return s.stats
    


def load_election(fname):
    """Load a pickled Election object from file"""
    with open(fname, 'rb') as f:
        e = pickle.load(f)
    return e





    
def __plot1d(election, results, title=''):
    """Visualize election for 1-dimensional preferences
    
    Parameters
    ----------
    election : Election object
    
    results : list of ElectionResults
        Results of various election methods to compare
    """
    
    v = election.voters
    c = election.candidates

    markers = itertools.cycle(('o','v','^','<','>','s','*','+','P')) 
    
    h, edges = np.histogram(v, bins=20, density=True)
    bin_centers = .5*(edges[0:-1] + edges[1:])

    # create plot for candidate preferences
    yc = np.interp(c.ravel(), bin_centers, h, )
    
    fig, ax = plt.subplots()
    ax.plot(bin_centers, h, label='voter distribution')    
    ax.plot(c, yc, 'o', ms=10, fillstyle='none', label='candidates')
    
    # create plot for winner preferences
    for result in results:
        w = result.winners
        cw = c[w]    
        yw = np.interp(cw.ravel(), bin_centers, h, )    
#        ax.plot(cw, yw, ms=10, marker=next(markers), label=result.methodname)
        ax.plot(cw, yw, ms=6.5, marker=next(markers), label=result.methodname)
#        ax.annotate(result.methodname, (cw, yw + .01))

    ax.set_xlabel('Voter Preference')
    ax.set_ylabel('Voter Population Density')

    mean = election.stats.mean_voter
    median = election.stats.median_voter
    ymean = np.interp(mean, bin_centers, h,)
    ymedian = np.interp(median, bin_centers, h,)    

    ax.plot(mean, ymean, '+', label='mean')
    ax.plot(median, ymedian, 'x', label='median')    
    plt.legend()
    plt.grid()
    plt.title(title)
    # create plot of regrets for all possible 1-d candidates within 2 standard deviations
    arr1 = np.linspace(bin_centers[0], bin_centers[-1], 50)
    r = metrics.candidate_regrets(v, arr1[:, None])
    ax2 = ax.twinx()
    ax2.plot(arr1, r, 'r', label='Pref. Regret')
    ax2.set_ylabel('Voter Regret')
    ax2.set_ylim(0, None)
    plt.legend()
    

    
def _plot_hist(output):
    """
    Plot histogram information from output from `simulate_election`
    """
    edges = output['h_edges']
    
    
    xedges = 0.5 * (edges[0:-1] + edges[1:])
    voters = output['h_voters'] 
    candidates = output['h_candidates']
    winners = output['h_winners']
    print(winners)
    plt.plot(xedges, voters, label='voters')
    plt.plot(xedges, candidates, 'o-', label='candidates')
    plt.plot(xedges, winners, 'o-', label='winners')
    plt.legend()



def __plot2d(election, results, title=''):

    v = election.voters
    c = election.candidates
    markers = itertools.cycle(('o','v','^','<','>','s','*','+','P')) 
    
    h, xedges, yedges = np.histogram2d(v[:,0], v[:,1], bins=20, normed=True)
    xbin_centers = .5*(xedges[0:-1] + xedges[1:])
    ybin_centers = .5*(yedges[0:-1] + yedges[1:])
    
    fig = plt.figure(figsize=(12, 8))
#    plt.pcolormesh(xbin_centers, ybin_centers, h,)
    plt.contourf(xbin_centers, ybin_centers, h, 20)
    plt.plot(c[:,0], c[:,1], 'o', ms=10, label='candidates')
    
    for result in results:
        w = result.winners
        cw = c[w]
        plt.plot(cw[:,0], cw[:, 1],
                 ms=6.5,
                 marker=next(markers), 
                 label=result.methodname)
    
    plt.xlabel('Voter Preference 0')
    plt.ylabel('Voter Preference 1')    
    mean = election.stats.mean_voter
    median = election.stats.median_voter
    plt.plot(mean[0], mean[1], '+', label='mean')
    plt.plot(median[0], median[1], 'x', label='median')        
    plt.legend()
    
    #
        
