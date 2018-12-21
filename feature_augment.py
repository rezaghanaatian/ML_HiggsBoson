# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def build_squre_cross_terms(x, degree, ct):
    n_x = len(x)
    nbr_param = len(x[0])
    #print(nbr_param)
    nbr_ct = 0
    nbr_ct += int(nbr_param * (nbr_param - 1) / 2)
    mat = np.zeros((n_x, (degree + 1) * nbr_param + nbr_ct))
    
    mat[:, :nbr_param] = x
	idx = nbr_param
	
	if degree = 1:
		mat[:, nbr_param+1:2*nbr_param] = np.square(x)
		idx = 2*nbr_param
		
	if degree = 2:
		mat[:, nbr_param+1:2*nbr_param] = np.square(x)
		mat[:, 2*nbr_param+1:3*nbr_param] = np.power(x,3)
		idx = 2*nbr_param
    
	if ct = 1:
		for l in range(nbr_ct):
			for m in range(l + 1, nbr_param):
				mat[:, idx] = x[:, l] * x[:, m]
				idx += 1
                
    return mat