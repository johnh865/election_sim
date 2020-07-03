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




