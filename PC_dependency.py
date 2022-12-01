#!/usr/bin/env python3

import deepdish as dd
import pandas as pd
#import dependency

Contdict = dd.io.load('PSPC_cont_tables.h5')
PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]

for subject,res_tmp in Contdict.items():
	res = res_tmp[PCcols]
	for pair in pair_array:
		 dep = dependency (res, pair)
		
	
	