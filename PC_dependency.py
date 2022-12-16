#!/usr/bin/env python3

import deepdish as dd
import pandas as pd
from dependency import dependency

Contdict = dd.io.load('PSPC_cont_tables.h5')
PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]

output = []
for subject,res_tmp in Contdict.items():
	res = res_tmp[PCcols]
	for pair in pair_array:
		dep = dependency (res, pair)
		output.append(
        {
            'Subject': subject,
            'Pair': pair,
			'Dependency Data': dep[0],
			'Dependency Independent Model': dep[1],
			'Dependency Dependent Model': dep[2]
        })
			
outputdf = pd.DataFrame(output)
	
	