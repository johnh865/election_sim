# -*- coding: utf-8 -*-

"""
Methods to postprocess election results
"""
import numpy as np
import pandas as pd

def categorize_condorcet(df: pd.DataFrame):
    """
    Categorize Elections based on Condorcet/Utility/Plurality/Majority 
    conditionals. Categories focus on Condorcet criterion. 
    
    Parameters
    ----------
    df : Pandas DataFrame
        Dataframe produced by :module:`~votesim.benchmarks`.
    
    Returns
    --------
    df : Pandas DataFrame
        Dataframe with new 'categories' column. 
        
    
    Category Combinations
    
    Labels 
    -------
    - M = majority winner
    - P = plurality winner that is not majority winner
    - C = condorcet winner that is not majority winner
    - U = utility winner
    
    Categories
    ----------
    - MU = Has majority utility winner
    - M = Has majority winner that is not utility winner.
    
    - CPU = Has condorcet, utility, plurality winner
    - CU = Has condorcet, utility winner that is not plurality winner
    - CP = Has condorcet, plurality winner that is not utility winner
    - C = Has condorcet winner who is not plurality and utility winner
    - NC = Has no Condorcet winner
    

    """
    
    iM = df['output.candidate.winner_majority']
    iP = df['output.candidate.winner_plurality']
    iC = df['output.candidate.winner_condorcet']
    iU = df['output.candidate.winner_utility']
    
    df = df.copy()
    df.loc[:, 'categories'] = 'No category'
    
    maj = iM > -1
    no_maj = ~maj
    
    MU = (iM == iU)
    M = maj & (iM != iU)
    
    CPU = no_maj & (iC == iP) & (iC == iU)
    CP  = no_maj & (iC == iP) & (iC != iU)
    CU  = no_maj & (iC == iU) & (iC != iP)
    #PU  = no_maj & (iP == iU) & (iP != iC)  # not mutually exclusive 
    NC = (iC == -1)    
    C = (iC > -1) & (iC != iP) & (iC != iU)
    
    df.loc[MU, 'categories'] = 'MU'
    df.loc[M, 'categories'] = 'M'
    df.loc[CPU, 'categories'] = 'CPU'
    df.loc[CP, 'categories'] = 'CP'
    df.loc[CU, 'categories'] = 'CU'
    df.loc[C, 'categories'] = 'C'
    #df.loc[PU, 'categories'] = 'PU'
    df.loc[NC, 'categories'] = 'nc'
    
    return df


def benchmark_score(df: pd.DataFrame):
    """Construct benchmark score of how good electoral methods are.
    
    """
    xname1 = 'output.winner.regret_efficiency_voter'
    xname2 = 'output.winner.regret_efficiency_candidated'
    eff1 = df[xname1]
    eff2 = df[xname2]
    regret = 1 - np.minimum(eff1, eff2)
    pratio = df['output.candidate.plurality_ratio'] * 100
    
    