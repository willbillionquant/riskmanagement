import os
codePath_betSim = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from itertools import product
from scipy.stats import norm

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def getSim_fixPctBet(initAmount=100, f=12.5, p=0.5, b=1.5, numTrial=50, numSim=400):
    """
    Obtain dictionary of equal-percent-bet simulation results.
    initAmount: initial amount for betting
    f: percent per bet
    p: winning probability
    numTrials: number of trials of the same binary game
    numSim: number of binary simulations
    """
    # Dict for recording different series of profit/loss
    dictSim = {}
    # Dict for recording different series of capital (total equity)
    dictAmount = {}
    # Generate a total of `numSim`= N series of binary game result and P/L
    for num in range(numSim):
        # Generate a series of "1"/"0" with probability p
        stepfunc = lambda x: 1 if x > 0 else 0  # Lambda Function for getting "1" with a fixed prob and "0" otherwise
        arrSign = np.array([stepfunc(r) for r in np.random.uniform(p - 1, p, numTrial)])
        # Series of asset growth factor based on win/loss result (EITHER (1 + b * f%) OR (1 -f%))
        arrTrial = 1 + f * ((b + 1) * arrSign - 1) / 100
        # Record the asset growth factor series into `dictSim`
        dictSim[num + 1] = arrTrial
        # Array of total equity and record into `dictAmount`
        dictAmount[f's{num + 1}'] = initAmount * arrTrial.cumprod()
    # Form dataframe from the `dictAmount` and transpose, so that each row corresponds to a betting series
    dfSim = pd.DataFrame(dictAmount).transpose()
    # Rename columns so that each number in column labels corresponds to the k-th trial
    dfSim = dfSim.rename(columns={k: (k + 1) for k in dfSim.columns})

    return dfSim

def getSim_fixLev(initAmount=100, lev=1.00, miu=0.05, sig=0.2, numPeriod=60, numSim=1000):
    """
    Obtain dataframe of fixed-leverage-bet simulations, with returns of each interval normally distributed.
    Assume zero-cost-rebalance at the end of each period.
    initAmount: initial capital
    miu: (non-annualized) mean return
    sig: (non-annualized) sigma
    numPeriod: number of periods
    numSim: number of simulations
    """
    # Dict for recording different series of total equity
    dictAmount = {}
    # Generate a total of `numSim`= N series of normally distributed returns
    for num in range(numSim):
        # vector of log returns in each period and exponentiate
        arrPct = np.exp(np.random.normal(miu, sig, numPeriod))
        # convert into growth factor vector by converting to percentage change vector, multiply by leverage, and add 1
        arrFactor = 1 + lev * (arrPct - 1)
        # equity vector by cumulative multiplying by growth factors
        arrAmount = initAmount * arrFactor.cumprod()
        # IF equity drops to 0 or even below (due to over-leverage), stop betting, set the remaining equity to 1/10000
        # of initial amount and fix it in the remaining series (for the sake of legal semi-log equity curve plotting)
        # (This artificial "residual equity" is unreal assummption, the reality is more cruel than this!)
        numBet = 1
        amtRuin = initAmount / 10000
        while numBet <= numPeriod - 1:
            if arrAmount[numBet] <= amtRuin:
                for j in range(numBet, numPeriod):
                    arrAmount[j] = amtRuin
                break
            numBet += 1
        dictAmount[f's{num + 1}'] = arrAmount
        # Form dataframe from the `dictAmount` and transpose, so that each row corresponds to a betting series
    dfSim = pd.DataFrame(dictAmount).transpose()
    # Rename columns so that each number in column labels corresponds to the k-th trial
    dfSim = dfSim.rename(columns={k: (k + 1) for k in dfSim.columns})

    return dfSim

def getProfitfactor(p, b):
    """Get profit factor of a binary game."""
    return round(p * b / (1 - p), 4)

def getOdds(p, pf):
    """Given fixed winning rate and profit factor, find odds (or reward-risk ratio)."""
    return pf * (1 - p) / p

def getWinrate(b, pf):
    """Given fixed odds and profit factor, find winning rate."""
    return pf / (pf + b)

def getSimKPI_fixPctBet(initAmount=100, f=12.5, p=0.5, b=1.5, numTrial=50, numSim=400):
    """Obtain a dictionary of final performance KPI of simulations. """
    dfSim = getSim_fixPctBet(initAmount, f, p, b, numTrial, numSim)
    dictKPI = {}
    dictKPI['p'] = p
    dictKPI['b'] = b
    dictKPI['f'] = f
    dictKPI['profitfactor'] = round(getProfitfactor(p, b), 4)
    dictKPI['win%'] = round(100 * dfSim[dfSim[numTrial] >= initAmount].shape[0] / numSim, 2)
    dictKPI['amountAvg'] = round(dfSim[numTrial].mean(), 2)
    dictKPI['amountMed'] = round(dfSim[numTrial].median(), 2)
    dictKPI['amountStd'] = round(dfSim[numTrial].std(), 2)

    return dfSim, dictKPI
