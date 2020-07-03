# Election Sim

Election simulator for comparing plurality with score, instant-runoff, 
condorcet methods, and others. 

## Features
- numpy-style array operations
- N-dimensional, spatial voter preference model
- Single and multiwinner election methods
- Run election method benchmarks to gauge performance


## Usage Guide
- Example scripts scan be found in /notebooks/ folder.
- Python importable module found at /votesim/


## Votesim Package Contents
- Various voting systems implemented in votesim.votesystems
    - Single Winner
         - Scored methods
         - Condorcet Smith minimax
         - Condorcet Ranked Pairs
    - Multi-winner methods
         - Reweighted range
         - Sequential Monroe
         - Single Transferable Vote
- Voter models found in votesim.models
- Voter evaluation metrics found in votesim.metrics

### Votesim Benchmarks

Benchmarks combine the voter, candidate, and election model
as well as assumed parameters for the models in order to assess
any inputed voting system. 

# Installation for Windows for Dummies

This installation guide is written for non-developers and hobbyists. 

The first step is to download this package and extract it somewhere 
on your computer. 

Then you probably need to install the Python Anaconda
distribution found in https://www.anaconda.com . Download and install. 

After installing Anaconda, open "Anaconda Prompt" which is probably
located in your Window's Start Menu. In this prompt, navigate to 

    cd ...\election_sim\scripts\

where "..." is the directory where you've copied election_sim. 
Then run python script within Anaconda Prompt:

    python setup_env.py

This script configures the environment. Close Anaconda Prompt. You can now
directly open the following scripts from Windows Explorer:

```
install2conda.bat
jupyter.bat
```
- install2conda.bat -- Installs votesim to Python Anaconda's site packages.
- jupyter.bat -- Launches Jupyter Notebook

Jupyter Notebook is launched at the votesim directory, and you can
explore some of the results of the votesim simulations. 


# Using Jupyter Notebooks

## Opening Jupyter Notebook

After setup, you can directly click on the file `jupyter.bat` found in \scripts\
from Windows to open a notebook.

## Imports 

In order to use votesim in Jupyter, you must first import `init_votesim.py` to set up some path configurations.
In your notebooks, use: 

    import init_votesim

This just tells Jupyter where votesim is located,
and it also tells Jupyter where to save output data in path `datapath`.

Visit https://jupyter.org/ for more information.  

Now you're all set and you can explore the examples in /notebooks/ for more information on how to use votesim.






