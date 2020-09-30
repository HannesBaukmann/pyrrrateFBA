#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:10:06 2020

@author: markukob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Solutions:#(pd.DataFrame):
    """
    Collector/Container for results of dynamic simulations
    # MAYBE: inherit from pandas DataFrame instead of copying?
    TODO - extend/delete elements, interpolate etc.
         - Kann es passieren, dass eines der Felder leer ist? -> Edge case handling
    """
    def __init__(self, tt, tt_shift, sol_y, sol_u, sol_x):
        #super().__init__()
        self.dyndata = pd.DataFrame() #
        self.dyndata['time'] = tt
        self.dyndata.set_index('time', inplace=True) # MAYBE: This could be done in one step 
        self.condata = pd.DataFrame()
        self.condata['time_shifted'] = tt_shift
        self.condata.set_index('time_shifted', inplace=True)
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


    def extend_y(self, tnew, y_new):
        """
        Add new data to the dynamic variables
        """
        for i, t in enumerate(tnew):
            # MAYBE: Additional safety measures: We don't check the y-values by name.
            self.dyndata.loc[t] = y_new[i,:] # QUESTION: Should we try to eliminate instances with times that are too close?
        # sort
        self.dyndata.sort_index(inplace=True)


    def extend_ux(self, tshiftnew, u_new, x_new):
        """
        Add new data to the dynamic variables
        """
        for i, t in enumerate(tshiftnew):
            # MAYBE: Additional safety measures: We don't check the u/x-values by name.
            self.condata.loc[t] = np.stack([np.array(u_new[i, :]),
                                            np.array(x_new[i, :])]).flatten() # QUESTION: Should we try to eliminate instances with times that are too close?
        # sort
        self.dyndata.sort_index(inplace=True)
        
        
    def plot_all(self, **kwargs): # Maybe: Bad name?
        """
        plot the data contained in the DataFrames into three figures
        TODO
        - output
        - filter values
        - control names/titles/scaling etc.
        """
        use_subplots = kwargs.get('subplots', False)
        y_data_indices = kwargs.get('y_data_indices', None)
        log_y = kwargs.get('logy', False)
        # plot dynamic data
        ax = None
        if use_subplots:
            ax = plt.subplot(3, 1, 1)
        self.dyndata.plot(#x='time', 
                          y=self._get_name_from_index(y_data_indices),
                          marker='.', logy=log_y, ax=ax)
        #plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
        plt.xlim(min(self.dyndata.index), max(self.dyndata.index))
        plt.xlabel('time')
        plt.title('Dynamic Data')
        plt.show()
        # plot control vectors
        u_names = [i for i in self.condata.keys() if 'u' in i]
        if u_names:
            ax = None
            if use_subplots:
                ax = plt.subplot(3, 1, 2)
            self.condata.plot(#x='time_shifted', 
                              y=u_names, marker='.', ax=ax)
            #plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
            plt.xlim(min(self.dyndata.index), max(self.dyndata.index))
            plt.xlabel('time')
            plt.title('Control Data')
            plt.show()
        # plot x control data
        x_names = [i for i in self.condata.keys() if 'x' in i]
        if x_names:
            ax = None
            if use_subplots:
                ax = plt.subplot(3, 1, 3)
            self.condata.plot(#x='time_shifted', 
                              y=x_names, marker='.', ax=ax)
            #plt.xlim(min(self.dyndata['time']), max(self.dyndata['time']))
            plt.xlim(min(self.dyndata.index), max(self.dyndata.index))
            plt.xlabel('time')
            plt.title('x Control Data')
            plt.show()

    def _get_name_from_index(self, index_list, part='dyn'):
        if part == 'dyn':
            if index_list:
                name_list = [self.dyndata.keys()[idx] for idx in index_list]
            else:
                name_list = list(self.dyndata.keys())
        else:
            if index_list:
                name_list = [self.condata.keys()[idx] for idx in index_list]
            else:
                name_list = list(self.condata.keys())
        return name_list


    def _get_index_from_name(self, name_list, part='dyn'):
        #list(self.dyndata.keys().index(name))
        if part == 'dyn':
            index_set = [self.dyndata.columns.get_loc(name) for name in name_list]
        else:
            index_set = [self.condata.columns.get_loc(name) for name in name_list]
        return index_set


    def _create_data_names(sol_y, sol_u, sol_x):
        dyndata_names = ['y'+str(i) for i in range(sol_y.shape[1])]
        condata_names = ['x'+str(i) for i in range(sol_x.shape[1])]
        condata_names += ['u'+str(i) for i in range(sol_u.shape[1])]
        return dyndata_names, condata_names

