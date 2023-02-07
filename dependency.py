#!/usr/bin/env python3

import numpy as np

def dependency(res,pair,guess=1,c=4):
	#function to calculate behavioural dependency measure
	# Modified version of Matlab function from Aidan J Horner 04/2015

	# input:
	# res       =   MxN matrix of M 'events' and N 'retrieval trials'
	# pair      =   pair of retrieval trials to calculate dependency 
	#               (e.g., cue location retrieve object and cue location retrieve person)
	#               example = [1 2] - uses first two columns to build contigency table for analysis
	# optional inputs:
	# guess     =   include guessing in Dependent Model (1 or 0)
	#               default = 1
	# c         =   number of choices (i.e., c-alternative forced choice) - for
	#               estimating level of guessing
	#               default = 4

	# output:
	# dep       =   dependency measure for [data independent_model dependent_model]
	dep = np.zeros((3))
	# calculate dependency for data
	res2 = res[pair]
	dep[0]      = sum(res2.sum(axis=1)==2)/len(res2) # calculate dependency for data

	# calculate dependency for independent model

	acc         = res2.mean() # calculate accuracy for each retrieval type
	dep[1]      = acc[0]*acc[1] # calculate dependenct for independent model

	# calculate dependency for dependent model

	cont        = np.empty((len(res2),2,2))*np.nan # create matrix for dependent model probabilities
	g           = (1-res.values.mean())*(c/(c-1)) # calculate level of guessing
	b           = res.mean(axis=0); b[pair] = np.nan # calculate average performance
	for i, a in res.iterrows(): # loop through all event   
		a[pair] = np.nan  # calculate event specific performance
		E       = a.mean()/b.mean() # calculate ratio of event / average performance (episodic factor)
		P = np.zeros((2))*np.nan
		for ii,p in enumerate(pair):
			if E*acc[p]>1:
				P[ii] = 1
			else:
				if guess == 1:
					P[ii] = (E*(acc[p]-(g/c)))+(g/c)
				elif guess == 0:
					P[ii] = E*acc[p]
		cont[i,0,0] = P[0]*P[1]
		cont[i,0,1] = (1-P[0])*P[1]
		cont[i,1,0] = P[0]*(1-P[1])
		cont[i,1,1] = (1-P[0])*(1-P[1])

	cont2       = sum(cont) # create contingency table
	dep[2]      = (cont2[0,0]+cont2[1,1])/np.sum(cont2) # calculate dependency for dependent model
	return dep
