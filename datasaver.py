# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:59:42 2021

@author: fmcpa
"""

import datetime
import pickle


def save(data, n_expe):
    now = datetime.datetime.now()
    dt_string = 'expe' + str(n_expe) + '-' + now.strftime("%d-%m-%Y-%H-%M-%S")
    
    file = open(dt_string, 'wb')
    pickle.dump(data, file)
    file.close()


    
def load(filename):
    file = open(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data