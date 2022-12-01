#!/usr/bin/env python3

import numpy as np

def dependency(res2,pair,guess=1,c=4):
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
	
	# calculate dependency for data
	res2 = res[pair]
	dep_1      = sum(res2.sum(axis=1)==2)/len(res2) # calculate dependency for data

	# calculate dependency for independent model

	acc        = res2.mean() # calculate accuracy for each retrieval type
	dep_2      = acc[0]*acc[1] # calculate dependenct for independent model

	# calculate dependency for dependent model

	cont        = nan(size(res2,1),2,2) # create matrix for dependent model probabilities
	g           = (1-res.values.mean())*(c/(c-1)) # calculate level of guessing
	b           = np.mean(res); b(:,pair) = nan # calculate average performance
	for i, a in res.iterrows(): # loop through all event   
		a(:,pair) = nan;                    # calculate event specific performance
		E       = nanmean(a)/nanmean(b);    # calculate ratio of event / average performance (episodic factor)
		for p = 1:2;
			if E*acc(p)>1
				P(p) = 1;
			else
				if guess == 1
					P(p) = (E*(acc(p)-(g/c)))+(g/c);
				elseif guess == 0
					P(p) = E*acc(p);
				end
			end
		end
		cont(i,1,1) = P(1)*P(2);
		cont(i,1,2) = (1-P(1))*P(2);
		cont(i,2,1) = P(1)*(1-P(2));
		cont(i,2,2) = (1-P(1))*(1-P(2));
	end
	cont2       = squeeze(sum(cont));                           % create contingency table
	dep(3)      = (cont2(1,1)+cont2(2,2))/sum(cont2(:));        % calculate dependency for dependent model
