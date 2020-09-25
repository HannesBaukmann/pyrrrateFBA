#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:10:06 2020

@author: markukob
"""

import pandas as pd
import matplotlib.pyplot as plt

class Solutions:#(pd.DataFrame):
    """
    Collector/Container for results of dynamic simulations
    # MAYBE: inherit from pandas DataFrame instead of copying?
    """
    def __init__(self, tt, tt_shift, sol_y, sol_u, sol_x):
        #super().__init__()
        self.dyndata = pd.DataFrame() #
        self.dyndata['time'] = tt
        self.condata = pd.DataFrame()
        self.condata['time_shifted'] = tt_shift
        #
        for i in range(sol_y.shape[1]):
            self.dyndata['y'+str(i)] = sol_y[:,i]
        for i in range(sol_x.shape[1]):
            self.condata['x'+str(i)] = sol_x[:,i]
        for i in range(sol_u.shape[1]):
            self.condata['u'+str(i)] = sol_u[:,i]

    def __str__(self):
        return 'Solutions object with dynamics data : \n' \
            + self.dyndata.__str__() + '\n\n' \
            + 'and control data : \n' \
            + self.condata.__str__()
    #    #return str(self.__class__) + ": " + str(self.__dict__) # A very generic print
    
    def plot_all(self, **kwargs):
        """
        plot the data contained in the DataFrames into three figures
        TODO
        - output
        - filter values
        - control names/titles/scaling etc.
        """
        # plot dynamic data
        self.dyndata.plot(x='time', marker='.')
        plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
        plt.xlabel('time')
        plt.title('Dynamic Data')
        plt.show()
        # plot control vectors
        u_names = [i for i in self.condata.keys() if 'u' in i]
        self.condata.plot(x='time_shifted', y=u_names, marker='.')
        plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
        plt.xlabel('time')
        plt.title('Control Data')
        plt.show()
        # plot x control data
        # TODO: Only if existent
        x_names = [i for i in self.condata.keys() if 'x' in i]
        self.condata.plot(x='time_shifted', y=x_names, marker='.')
        plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
        plt.xlabel('time')
        plt.title('x Control Data')
        plt.show()
