# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def vset():
    """Set votesim preferred matplotlib global options"""
    sns.set()
    mpl.rcParams['figure.figsize'] = [8, 6]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['font.size'] =  10
    
    subplot = {
          'figure.subplot.bottom': 0.1,
          'figure.subplot.hspace': 0.35,
          'figure.subplot.left': 0.1,
          'figure.subplot.right': 0.95,
          'figure.subplot.top': 0.9,
          'figure.subplot.wspace': 0.2, 
          }
    mpl.rcParams.update(subplot)
    return


def subplot_4set(**kwargs):
    figsize = [9, 6.5]
    fig = plt.figure(figsize=figsize, **kwargs)
    
    left = 0.05  # the left side of the subplots of the figure
    right = 0.95   # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9     # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
                  # expressed as a fraction of the average axis width
    hspace = .35  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height
    
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)
    return fig
      



def subplot_2set(**kwargs):
    figsize = [9.0, 3.5]
    fig = plt.figure(figsize=figsize, **kwargs)
    
    left = 0.1  # the left side of the subplots of the figure
    right = 0.98   # the right side of the subplots of the figure
    bottom = 0.2  # the bottom of the subplots of the figure
    top = 0.85     # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
                  # expressed as a fraction of the average axis width
    hspace = .35  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height            
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)
    return fig



def subplot_2row(**kwargs):
    figsize=[6.5, 9]
    fig = plt.figure(figsize=figsize, **kwargs)
    
    left = 0.1125  # the left side of the subplots of the figure
    right = 0.98   # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.925     # the top of the subplots of the figure
    wspace = 0.1  # the amount of width reserved for space between subplots,
                  # expressed as a fraction of the average axis width
    hspace = .3  # the amount of height reserved for space between subplots,
                  # expressed as a fraction of the average axis height            
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)
    return fig
    

def heatmap(x, y, hue, data,
            func='mean',
            xbin=None,
            ybin=None,
            annot=True,
            fmt='.1f',
            xfmt='g',
            yfmt='g',
            cbar=False,
            linewidths=0.5,
            cmap='viridis_r',
            sort=True,
            xsortkey=None,
            ysortkey=None,
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
    
    if ysortkey is not None:
        dfp = dfp.loc[ysortkey]
    if xsortkey is not None:
        dfp = dfp[xsortkey]

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
        data.loc[:, x] = xcat
    else:
        xbins = None
        data.loc[:, x] = data[x].astype('category')
        
    if ybin is not None:
        ycat, ybins = pd.cut(data[y], ybin, retbins=True)
        data.loc[:, y] = ycat
    else:
        ybins = None
        data.loc[:, y] = data[y].astype('category')
    
    
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



def bar(x, y, data=None, fmt='g', **kwargs):
    """
    Custom bar plot with values on bars
    
    Parameters
    ----------
    x : str
        data column name for x-axis
    y : str
        data column name for y-axis
    dataDataFrame, array, or list of arrays, optional
        Dataset for plotting.  
        
    Returns
    ---------
    ax : matplotlib Axes
        Returns the Axes object with the plot drawn onto it.
    """

    
    ax = sns.barplot(x=x, y=y, data=data, **kwargs)
    show_values_on_bars(ax, fmt=fmt)
    #num = len(x)
    #x1 = np.arange(num) - .5
    #fmt = '%' + fmt
    #yrange = np.max(y) - np.min(y)
    #ydelta = yrange / 25
    # for (xi, yi) in zip(x1, y):    
    #     s = fmt % yi
    #     ax.annotate(s, xy=(xi +.125, yi + ydelta))
    return ax


def show_values_on_bars(axs, height=.2, fmt='g'):
    """Put labels on seaborn bar chart from stack overflow
    https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
    
    
    """
    ffmt = '{:' + fmt + '}'
    
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + height
            value = ffmt.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

        


def auto_num_fmt(self, number, sf=3):
    
    
    
    if .001 < number < 10000 :
        fmt = '.' + str(sf) + '.f'
    else:
        fmt = '.' + str(sf) + '.e'
        
        
    
    

### test


def test_heat_category():
    x = np.random.randint(0, 5, size=1000)
    y = np.random.randint(0, 5, size=1000)
    z = x * y
    
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)
    heatmap('x', 'y', 'z', df, xfmt='.2f')
    return


def test_heat_continuous():
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    z = x * y
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)    
    heatmap('x', 'y', 'z', df, 
            xbin=10, ybin=10, yfmt='.2f', xfmt='.2f')
    
    
def test_heat_mixed():
    x = np.random.rand(1000)
    y = np.random.randint(0, 5, size=1000)
    z = x * y
    d = {'x': x,
         'y': y,
         'z': z,}
    df = pd.DataFrame(data=d)
    heatmap('x', 'y', 'z', df, xbin=10, xfmt='.2f')
    
    
def test_bar():
    
    plt.figure()
    x = np.arange(10)
    y = x + 6
    bar(x, y)
    ax = plt.gca()
    
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


def test_2set():
    
    fig = subplot_2set()
    plt.subplot(1,2,1)
    test_heat_continuous()
    plt.title('Subplot #1')
    plt.subplot(1,2,2)
    test_heat_mixed()
    plt.title('Subplot #2')
    plt.suptitle("this is the test title")
    
    
    
def test_2row():
    fig = subplot_2row()
    plt.subplot(2,1,1)
    test_heat_continuous()
    plt.title('Subplot #1')
    plt.subplot(2,1,2)
    test_heat_mixed()
    plt.title('Subplot #2')
    plt.suptitle("this is the test title")
    
    
if __name__ == '__main__':
    vset()
    test_2row()
    test_2set()

    
    # plt.figure()
    test_heat_mixed()
    # test_bar()