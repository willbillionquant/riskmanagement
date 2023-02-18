import os
codePath_betSim = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from itertools import product
from scipy.stats import norm

import plotly.express as px
import plotly.graph_objects as go

from riskmanagement.betsim import *

def getExpGrowth(p=0.5, b=1.5, f=0.1, n=1):
    """
    Obtain expected geometric return of repeated trials of a binary game.
    p: winning rate
    b: odds / reward-risk-ratio
    f: fixed percent of each bet
    n: number of trials
    """
    logGrowth = p * np.log(1 + b * f) + (1 - p) * np.log(1 - f)

    return np.exp(n * logGrowth)

def getNormalGrowth(lev=1.00, miu=0.03, sig=0.15, n=1):
    """Obtain expected geometric returns of random walk returns."""
    logGrowth = lev * (miu - sig ** 2 * lev / 2)

    return np.exp(n * logGrowth)

def plotExpGrowth(p=0.5, b=1.5, n=1, fMin=0, fMax=0.5, step=0.01):
    """Plot expected geometric growth given fixed winning rate & odds, and identify optimal f%."""
    # Form pandas Series of expected return of varying f%
    arrFpct = np.arange(fMin, fMax, step)
    dictGrowth = {f: getExpGrowth(p, b, f, n) - 1 for f in arrFpct}
    dfGrowth = pd.DataFrame(pd.Series(dictGrowth))
    # Plot interactive diagram of f-percent curve
    fig = px.line(x=dfGrowth.index, y=dfGrowth[0], labels={'x': 'f%', 'y': f'%return on {n} trials'})
    fig.add_trace(go.Scatter(x=dfGrowth.index, y=np.repeat(0, len(arrFpct)), name='0%'))
    # Identify optimal f%
    bestF = dfGrowth[0].idxmax()
    bestGrowth = dfGrowth[0].max()
    fig.add_trace(go.Scatter(x=(bestF,), y=(bestGrowth,), line_color='green', name='Opt-f%', mode='markers+text',
                             marker_size=10, text=f'{bestF, round(bestGrowth, 4)}', textposition='bottom center'))
    # Show diagram
    fig.show()

def plotNormalGrowth(miu=0.03, sig=0.15, n=1, levMin=0.20, levMax=4.0, step=0.001):
    """Plot expected geometric growth given miu & sigma, and identify optimal leverage."""
    # Form pandas Series of expected return of varying leverage
    arrLev = np.arange(levMin, levMax, step)
    dictGrowth = {lev: getNormalGrowth(lev, miu, sig, n) for lev in arrLev}
    dfGrowth = pd.DataFrame(pd.Series(dictGrowth))
    # Plot interactive diagram of f-percent curve
    fig = px.line(x=dfGrowth.index, y=dfGrowth[0], labels={'x': 'lev', 'y': f'growth factor on {n} periods'})
    fig.add_trace(go.Scatter(x=dfGrowth.index, y=np.repeat(0, len(arrLev)), name='0%'))
    # Identify optimal leverage
    bestLev = round(dfGrowth[0].idxmax(), 4)
    bestGrowth = round(dfGrowth[0].max(), 4)
    fig.add_trace(go.Scatter(x=(bestLev, ), y=(bestGrowth, ), line_color='green', name='Opt-lev', mode='markers+text',
                             marker_size=10, text=f'{bestLev, round(bestGrowth, 4)}', textposition='bottom center'))
    # Title
    fig.update_layout(title=f'Expected geometric growoth of N({miu}, {sig})', title_x=0.5, width=1000, height=500)
    # Show diagram
    fig.show()

def getKellyF(p, b):
    """Obtain the optimal f% by Kelly formula."""
    return round(max((p * b - 1 + p) / b, 0), 4)

def getkellyLev(miu, sig):
    """Obtain Kelly formula of optimal leverage."""
    return round(max(miu / sig**2, 0), 4)


def resampleOHLC(dfData, listSymbol, freq='M'):
    """Resample the OHLC dataframe into desired timeframe."""
    dictRule = {}
    for symbol in listSymbol:
        dictSymbol = {f'{symbol}_op': 'first', f'{symbol}_cl': 'last',
                      f'{symbol}_hi': 'max', f'{symbol}_lo': 'min', f'{symbol}_vol': 'sum'}
        dictRule.update(dictSymbol)

    dfData1 = dfData.resample(rule=freq, label='right').agg(dictRule)

    return dfData1



# Notebook
dfAll = getYahooData(listSymbol, True, strStart, strEnd)
dfAll_month = resampleOHLC(dfAll, listSymbol)

numMonth = 36
strStart1 = dfAll_month.index[-1 - numMonth].strftime('%Y-%m-%d')
strEnd1 = dfAll_month.index[-1].strftime('%Y-%m-%d')

dfOptLev = pd.DataFrame(columns=['miu', 'sig', 'NAsharpe', 'optlev'])
dfPct = pd.DataFrame()

for symbol in listSymbol:
    dfMonth = dfAll_month.loc[strStart1:strEnd1, [f'{symbol}_cl']]
    dfPct[f'{symbol}_pct'] = np.round(np.log(dfMonth[f'{symbol}_cl'] / dfMonth[f'{symbol}_cl'].shift(1)), 5)
    dfPct[f'{symbol}_chg'] = np.round(dfMonth[f'{symbol}_cl'] / dfMonth[f'{symbol}_cl'].shift(1) - 1, 5)
    dfOptLev.loc[symbol, 'miu'] = round(dfPct[f'{symbol}_pct'].mean(), 5)
    dfOptLev.loc[symbol, 'sig'] = round(dfPct[f'{symbol}_pct'].std(), 5)
    dfOptLev.loc[symbol, 'NAsharpe'] = round(
        dfOptLev.loc[symbol, 'miu'] / dfOptLev.loc[symbol, 'sig'] * (numMonth ** 0.5), 4)
    dfOptLev.loc[symbol, 'optLev'] = getkellyLev(dfOptLev.loc[symbol, 'miu'], dfOptLev.loc[symbol, 'sig'])

dfNAV_1x = pd.DataFrame(index=dfPct.index)

for symbol in listSymbol:
    dfNAV_1x[f'{symbol}_NAV'] = (1 + dfPct[f'{symbol}_chg']).cumprod()

dfNAV_1x.iloc[0] = 1.00

for symbol in listSymbol:
    dfNAV_1x[f'{symbol}_DD'] = dfNAV_1x[f'{symbol}_NAV'] / dfNAV_1x[f'{symbol}_NAV'].cummax() - 1
    dfNAV_1x[f'{symbol}_MDD'] = dfNAV_1x[f'{symbol}_DD'].cummin()

productfield = product(listSymbol, ['NAV', 'DD', 'MDD'])
dfNAV_1x = dfNAV_1x[[f'{symbol}_{field}' for symbol, field in productfield]]


for symbol, field in product(listSymbol, ['NAV', 'MDD']):
    dfOptLev.loc[symbol, f'{field}_1x'] = dfNAV_1x.loc[strEnd1, f'{symbol}_{field}']