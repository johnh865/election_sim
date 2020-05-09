# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def vset():
    """Set votesim preferred matplotlib global options"""
    sns.set()
    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 150
    


def heatmap(x, y, hue, data,
            func='mean',
            xbin=None,
            ybin=None,
            annot=True,
            fmt='.2f',
            xfmt='.2f',
            yfmt='.2f',
            cbar=False,
            linewidths=0.5,
            cmap='viridis_r',
            sort=True,
            **kwargs):
    """Custom heatmap for either categorical or numeric data.
    
    Parameters
    ------------
    x : str
        column name of data plotted on x-axis
    y : str
        Column name of data plotted on y-axis
    hue : str
        Column name of data values plotted on heatmap.
    data : Pandas DataFrame
        Data used in plot
    func : str or function
        aggregation function for `data.agg`, for example
        
        - 'min', 'mean', 'max', 'sum'
        - np.mean
        
    xbin, ybin : None, int, or array (n,)
    
        If x or y is not categorical data, bin
        
        - None (default) -- if x or y is categorical data, do nothing. 
        - int -- Set to number of bins to divide data using pandas.cut
        - array -- User defined bins to divide data using pandas.cut
        
    xfmt, yfmt : str
        Formatting string for x and y axes, default '.2f'.
    sort : bool
        Sort the results by their average across the x-axis. Default True. 
    
    **kwargs : 
        Additional arguments passed into `seaborn.heatmap`.
        
    Returns
    ------
    ax : matplotlib Axes
        Axes object with the heatmap.
    dfp : pandas DataFrame
        Pivot table used to construct heatmap
    """
    
    xfmt = '%' + xfmt
    yfmt = '%' + yfmt

    dfp, xbin, ybin = heat_pivot(data, x, y, hue, 
                                 func=func, xbin=xbin, ybin=ybin, sort=sort)

    ax = sns.heatmap(
                dfp,
                annot=annot,
                fmt=fmt,
                cbar=cbar,
                linewidths=linewidths,
                cmap=cmap,
                **kwargs
                )
    
    if xbin is not None:
        ax.set_xticks(np.arange(len(xbin)))        
        xlabels = [xfmt % xi for xi in xbin]
        ax.set_xticklabels(xlabels)
        
    if ybin is not None:
        ax.set_yticks(np.arange(len(ybin)))        
        ylabels = [yfmt % yi for yi in ybin]       
        ax.set_yticklabels(ylabels)
    return ax, dfp
    


def heat_pivot(data, x, y, hue,
               func='mean',
               xbin=None,
               ybin=None,
               sort=True,
               ):

    data = data.copy()
    if xbin is not None:
        xcat, xbins = pd.cut(data[x], xbin, retbins=True)
        data[x] = xcat
    else:
        xbins = None
        data[x] = data[x].astype('category')
        
    if ybin is not None:
        ycat, ybins = pd.cut(data[y], ybin, retbins=True)
        data[y] = ycat
    else:
        ybins = None
        data[y] = data[y].astype('category')
    
    
    data = data[[x, y, hue]]
    dfp = (data.groupby([x, y])
               .agg(func)
                .reset_index()
               .pivot(index=y, columns=x, values=hue)
           )
    
    if sort:
        regret_avg = np.nanmean(dfp.values, axis=1)
        ii = np.argsort(regret_avg)[::-1]
        sort_index = dfp.index[ii]
        dfp = dfp.loc[sort_index]

    return dfp, xbins, ybins





def test_heat_category():
    x = np.random.randint(0, 5, size=1000)
    y = np.random.randint(0, 5, size=1000)
    z = x * y
    
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)
    heatmap('x', 'y', 'z', df)
    return


def test_heat_continuous():
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    z = x * y
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)    
    heatmap('x', 'y', 'z', df, xbin=10, ybin=10)
    
    
def test_heat_mixed():
    x = np.random.rand(1000)
    y = np.random.randint(0, 5, size=1000)
    z = x * y
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)
    heatmap('x', 'y', 'z', df, xbin=10)
    
# def test_heat():
#     x = np.arange(10)
#     z = {}
#     z['a'] = x
#     z['b'] = x+2
#     z['c'] = x**1.1
#     z['d'] = -x + 3
#     z['e'] = -x + 4
#     z['f'] = -x + 5
#     df = pd.DataFrame(z)
#     heatmap(df)
#     plt.xlabel('test')
#     assert True

if __name__ == '__main__':
    plt.figure()
    test_heat_continuous()
    plt.figure()
    test_heat_category()
    plt.figure()
    test_heat_mixed()