import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import GLSLio

c = '#00425C'


    
    
    
def plot_Mtl_annual_minimum_qom_levels():
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15520-01-JAN-1900_slev.csv')
    
    ts_min = GLSLio.annual_min_qom_ts(ra)
    
    # Graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.subplots_adjust(bottom=.1, right=.95, left=.1)
    ts_min.plot(ax=ax, color=c, lw=1.5)
    ts_min.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)
    
    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')
    
    ylim = ax.get_ylim()
    dy = ylim[1] - ylim[0]
    
    ax.set_ylabel("Niveau [m]")
    ax.set_title("Minimum annuel du niveau d'eau à la jetée #1 du port de Montréal") 
    
    for y in [1934, 1965, 2001, 2007, 2012]:
        ax.annotate(str(y), (y, ts_min[y]+dy*.02), (y, ts_min[y]+dy*.8), ha='center', size='small', arrowprops=dict(arrowstyle='-', fc='w', ec='k', ), color='k')
    
    ax.annotate('Zéro\ndes cartes', (1920, 0), (1925, -.3), size=8, ha='center', \
        arrowprops=dict(arrowstyle='-', connectionstyle='angle3,angleA=90,angleB=0') )
    
    
def plot_Sorel_annual_minimum_qom_levels():
    ra = GLSLio.EC_H20('../data/Niveaux St-Laurent/15930-1-JAN-1916_slev.csv')
    
    ts_min = GLSLio.annual_min_qom_ts(ra)
    
    # Graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.subplots_adjust(bottom=.1, right=.95, left=.1)
    ts_min.plot(ax=ax, color=c, lw=1.5)
    ts_min.plot(style='o', ax=ax, ms=5, mec=c, mfc='w', mew=1)
    
    ax.grid(ls='-', lw=.1, color='#777777')
    ax.grid(axis='x')
     
    ylim = ax.get_ylim()
    dy = ylim[1] - ylim[0]
   
    ax.set_ylabel("Niveau [m]")
    ax.set_title("Minimum annuel du niveau d'eau à Sorel") 
    for y in [2001, 2007, 2012]:
        ax.annotate(str(y), (y, ts_min[y]+dy*.02), (y, ts_min[y]+dy*.8), ha='center', size='small', arrowprops=dict(arrowstyle='-', fc='w', ec='k'), color='k')

    ax.text(1961, 0, 'Zéro des cartes', size=8, ha='left', )

