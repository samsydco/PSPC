#!/usr/bin/env python3

# Compare PC and PS

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

figurepath = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
PSdf = pd.read_csv('PS.csv')
Dependencydf = pd.read_csv('Dependency.csv')
datadf = pd.read_csv('datadf.csv')

subj = datadf['Subject'].unique()
subj2 = Dependencydf['Subject'].unique()
subj3 = PSdf['Subject'].unique()
# Subject 2 is missing MDEM078 due to perfect accuracy!

output = []
for subj in subj2:
	tmp_demo = datadf[datadf['Subject']==subj]
	tmp_ps = PSdf[PSdf['Subject']==subj]
	tmp_pc = Dependencydf[Dependencydf['Subject']==subj]
	output.append({
		'Subject':subj,
		'Age':tmp_demo['Age'].values[0],
		'KBIT':tmp_demo['kbit_std_v'].values[0],
		'Delay':tmp_demo['Delay'].values[0],
		'Target Selection Rate':tmp_ps[tmp_ps['Selection']=='Target']['Proportion Selected'].iloc[0],
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
olsdf = outputdf.rename({'Target Selection Rate': 'PS_acc', 'Associative Recognition': 'PC_acc'}, axis='columns')
model_acc = ols("PC_acc ~ PS_acc + KBIT + Age + Delay", data=olsdf).fit()
print(model_acc.summary())
model_dep = ols("Dependency ~ PS_acc + KBIT + Age", data=olsdf).fit()
print(model_dep.summary())

# plot partial regression
from sklearn.linear_model import LinearRegression
nonadf = olsdf.dropna()
PS_regression = LinearRegression().fit(nonadf[["KBIT","Age","Delay"]], nonadf["PS_acc"])
PC_regression = LinearRegression().fit(nonadf[["KBIT","Age","Delay"]], nonadf["PC_acc"])
PS_residuals = nonadf["PS_acc"] - PS_regression.predict(nonadf[["KBIT","Age","Delay"]])
PC_residuals = nonadf["PC_acc"] - PC_regression.predict(nonadf[["KBIT","Age","Delay"]])
residualsdf = pd.DataFrame({'Pattern Completion Accuracy': PC_residuals, 'Pattern Seperation Accuracy': PS_residuals})
f, ax = plt.subplots(figsize=(6, 6))
g1 = sns.regplot(data=residualsdf,x="Pattern Seperation Accuracy", y="Pattern Completion Accuracy" ,ax=ax)
f.savefig(figurepath+'PS_vs_PS_given_KBIT_Age.png', bbox_inches='tight', dpi=100)
	
	


