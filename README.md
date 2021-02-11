# Robust and integrative Bayesian neural networks for likelihood-free parameter inference
Code repository for:

- F. Wrede, R. Eriksson, R. Jiang, L. Petzold, S. Engblom, A. Hellander, P. Singh. Robust and integrative Bayesian neural networks for likelihood-free parameter inference. 2021, arXiv: 

## Install dependencies
Inside a virutal environment:

```pip install -r requirements.txt```

## Usage
There are two exmples included:

- Moving average of order 2 (MA2)
- Lotka-volterra (LV), discrete and stochastic population model

Each folder has a couple of jupyter notebooks and scripts:

- experiments.ipynb: 
    
    Running parameter inference experiments using the proposed methodology Bayesian Convolutional NN (BCNN), SNPE-C and ABC-SMC

- abc_smc.ipynb:

    More interactive ABC-SMC parameter inference

- plotting.ipynb:

    Plots figure based on the results from above.
