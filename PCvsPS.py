#!/usr/bin/env python3

# Compare PC and PS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from partial_regression import partial_regression

year = 1

figurepath = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
PSdf = pd.read_csv('csvs/PS_Year_'+str(year)+'.csv')
PScatdf = pd.read_csv('csvs/PS_cat_Year_'+str(year)+'.csv')
Dependencydf = pd.read_csv('csvs/Dependency_Year_'+str(year)+'.csv')
datadf = pd.read_csv('csvs/datadf.csv')
datadf = datadf[datadf['Year'] == year]

subj = datadf['Subject'].unique()
subj2 = Dependencydf['Subject'].unique()
subj3 = PSdf['Subject'].unique()
# Subject 2 is missing MDEM078 due to perfect accuracy!

PScols = {'object':'Object Target Selection Rate','location':'Location Target Selection Rate','animal':'Animal Target Selection Rate'}

output = []
for subject in subj2:
	tmp_demo = datadf[datadf['Subject']==subject]
	tmp_ps = PSdf[PSdf['Subject']==subject]
	tmp_ps_cat = PScatdf[PScatdf['Subject']==subject]
	tmp_pc = Dependencydf[Dependencydf['Subject']==subject]
	output.append({
		'Subject':subj,
		'Age':tmp_demo['Age'].values[0],
		'KBIT':tmp_demo['kbit_std_v'].values[0],
		'Delay':tmp_demo['Delay'].values[0],
		'Target Selection Rate':tmp_ps[tmp_ps['Selection']=='Target']['Proportion Selected'].iloc[0],
		'Object Target Selection Rate':tmp_ps_cat['object target selection rate'].iloc[0],
		'Location Target Selection Rate':tmp_ps_cat['location target selection rate'].iloc[0],
		'Animal Target Selection Rate':tmp_ps_cat['animal target selection rate'].iloc[0],
		'Lure Selection Rate':tmp_ps[tmp_ps['Selection']=='Lure']['Proportion Selected'].iloc[0],
		'Foil Selection Rate':tmp_ps[tmp_ps['Selection']=='Foil']['Proportion Selected'].iloc[0],
		'Dependency':tmp_pc['Dependency'].values[0],
		'Associative Recognition':tmp_pc['Accuracy'].values[0]
		
	})

outputdf = pd.DataFrame(output)

# partial correlation heatmap
f, ax = plt.subplots(figsize=(6, 5))
partial_corr_array = outputdf.drop(['Subject'], axis=1).corr()
matrix = np.triu(partial_corr_array)
cmap = sns.diverging_palette(30, 10, as_cmap=True)
sns.heatmap(partial_corr_array, mask=matrix, cmap=cmap, vmax=1,vmin=-1, center=0,square=True, linewidths=1, cbar_kws={"shrink": 0.9}, annot=True)
f.savefig(figurepath+'heatmap.png', bbox_inches='tight', dpi=100)

# partial regression relating pc_acc to ps_acc
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
olsdf = outputdf.rename({'Target Selection Rate': 'PS_acc', 'Associative Recognition': 'PC_acc','Object Target Selection Rate':'object','Location Target Selection Rate':'location','Animal Target Selection Rate':'animal'}, axis='columns')
model_acc = ols("PC_acc ~ PS_acc + KBIT + Age + Delay", data=olsdf).fit()
print(model_acc.summary())
model_dep = ols("Dependency ~ PS_acc + KBIT + Age + Delay", data=olsdf).fit()
print(model_dep.summary())
model_pc_dep = ols("Dependency ~ PC_acc + KBIT + Age + Delay", data=olsdf).fit()
print(model_pc_dep.summary())
d = {'PSPC':model_acc,'PS-Dep':model_dep,'PC-Dep':model_pc_dep}
for k,model in d.items():
	var='PS_acc' if 'PS' in k else 'PC_acc'
	df = model.df_resid
	tval = model.tvalues[var]
	rval = np.sqrt(np.square(tval) / (np.square(tval) + df))
	pval = model.pvalues[var]
	print('For model: '+k+' r('+str(df)+')='+str(rval)+', p = '+str(pval))
for y in ['PC_acc','Dependency']:
	for cat in PScols.keys():
		model = ols(y+" ~ "+cat+" + KBIT + Age + Delay", data=olsdf).fit()
		print(model.summary())
		df = model.df_resid
		tval = model.tvalues[cat]
		rval = np.sqrt(np.square(tval) / (np.square(tval) + df))
		pval = model.pvalues[cat]
		print('For model '+y+' vs. PS-'+cat+': r('+str(df)+')='+str(rval)+', p = '+str(pval))

# plot partial regression
from sklearn.linear_model import LinearRegression
for y,y_ in {'PC_acc':'Pattern Completion Accuracy','Dependency':'Dependency'}.items():
	for x,x_ in dict({"PC_acc":'Pattern Completion Accuracy',"PS_acc":'Pattern Seperation Accuracy'}, **PScols).items():
		if x!=y:
			figname = figurepath+x+'_vs_'+y+'_given_KBIT_Age'+'.png'
			partial_regression(olsdf, x, y, x_, y_,["KBIT","Age","Delay"], False, figname)
			

	
	


