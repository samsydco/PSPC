#!/usr/bin/env python3

# Analyse PS
# break out targets, lures, and foils

import tqdm as tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deepdish as dd
import seaborn as sns
import math
from scipy import stats

year = 1

figurepath = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
datadf = pd.read_csv('csvs/datadf.csv')
Contdict = dd.io.load('csvs/PSPC_cont_tables.h5')
datadf = datadf[datadf['Year'] == year]


PScols = ['object','location','animal']
PSkey = {-1:'Foil',1:'Target',0:'Lure'}

output = []
percatoutput = []
for subject,tmp in tqdm.tqdm(Contdict[year].items()):
	PStmp = tmp[PScols].to_numpy()
	tmpdf = datadf[datadf['Subject']==subject]
	age = tmpdf['Age'].values[0]
	sex = tmpdf['Gender'].iloc[0]
	delay=tmpdf['Delay'].iloc[0]
	nitems = np.count_nonzero(~np.isnan(PStmp.astype(float))) # PStmp.size (if did both blocks)
	d_ = {'Subject': subject,'Delay':delay,'Age':age,'Sex':sex}
	d = d_.copy()
	for col in PScols:
		d[col+' target selection rate'] = np.count_nonzero(tmp[col] == 1) / (nitems/3)
	for value in PSkey.keys():
		d2 = d_.copy()
		d2['Selection'] = PSkey[value]
		d2['Proportion Selected']  = np.count_nonzero(PStmp == value) / nitems
		d[PSkey[value]] = d2['Proportion Selected']
		output.append(d2)
	percatoutput.append(d)
	
outputdf = pd.DataFrame(output)
outputdf.to_csv('csvs/PSoutputdf_exactage.csv',index=False)
percatoutputdf = pd.DataFrame(percatoutput)
outputdf.to_csv('csvs/PS_Year_'+str(year)+'.csv',index=False)
percatoutputdf.to_csv('csvs/PS_cat_Year_'+str(year)+'.csv',index=False)
outputdf['Age'] = outputdf['Age'].map(lambda age: math.floor(age))
delaydf = outputdf[(outputdf.Delay)]
nodelaydf = outputdf[(outputdf.Delay == False)]

outputdf.to_csv('csvs/PSoutputdf.csv',index=False)
delaydf.to_csv('csvs/PSdelaydf.csv',index=False)

order = ['Target','Lure','Foil']
orderage = [4,5,6,7]


# cannot run rmANOVA due to unequal group sizes (unequal number of subjects in each age group)
# Therefore running LMM
import statsmodels.formula.api as smf

outputdf['Age'] = outputdf['Age'].map(lambda age: math.floor(age))
outputdf = pd.DataFrame(output)
outputdf.rename(columns={'Proportion Selected': 'PS'}, inplace=True)
outputdf['Selection'] = outputdf['Selection'].replace(['Foil','Target','Lure'], [-1,1,0])
md = smf.mixedlm('PS~Age+Selection+Age*Selection',outputdf,groups=outputdf["Subject"])
mdf = md.fit()
print('Model with all possible items: target/lure/foil:')
print(mdf.summary())
for k,v in PSkey.items():
	outputdf = pd.DataFrame(output)
	outputdf.rename(columns={'Proportion Selected': 'PS'}, inplace=True)
	outputdf['Selection'] = outputdf['Selection'].replace(['Foil','Target','Lure'], [-1,1,0])
	outputdf = outputdf.drop(outputdf[outputdf.Selection == k].index)
	md = smf.mixedlm('PS~Age+Selection+Age*Selection',outputdf,groups=outputdf["Subject"])
	mdf = md.fit()
	print('Model without '+v)
	print(mdf.summary())

import pingouin as pg
outputdf = pd.DataFrame(output)
res = pg.rm_anova(dv='Proportion Selected',within='Selection',subject='Subject',data=outputdf)
post_hocs = pg.pairwise_tests(dv='Proportion Selected',within='Selection',subject='Subject',padjust='bonf',data=outputdf)


	
	


