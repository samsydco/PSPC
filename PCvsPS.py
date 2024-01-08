#!/usr/bin/env python3

# PC vs PS
# Each subject gets one point (average PC and average PS across all trials)

import pandas as pd
from scipy import stats
import statsmodels.formula.api as sm2
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme(style="white",font_scale=1.5)

PSdf = pd.read_csv('csvs/PS_cat_Year_1.csv')
PCdf = pd.read_csv('csvs/Dependency_Year_1.csv')
combodf = pd.merge(PCdf, PSdf, how='outer', indicator=True)
missingdf = combodf[combodf['_merge'].str.contains('only')]
if len(missingdf)==1 and missingdf['Subject'].iloc[0]=='MDEM127':
	print('MDEM127 Did not complete both blocks of experiment\n'+
	'And therefore has incomplete PC data')
	combodf = combodf.drop(combodf[combodf['Subject']=='MDEM127'].index)

# High/low performing PC subjects:
excludedf = PCdf[(PCdf['Accuracy']<0.3) | (PCdf['Accuracy']>0.95)]
exclude_subjs = excludedf['Subject']

combodf.rename(columns={'Target': 'Target Selection Rate', 'Lure': 'Lure Selection Rate', 'Foil': 'Foil Selection Rate', 'Target': 'PS Target Selection Rate'}, inplace=True)

for delay in [True,False]:
	for exclude in [True,False]:
		for ps in ['Target']:#ure','Foil']:
			for pc in ['Accuracy','Dependency']:
				tmp = combodf[combodf['Delay']==delay]
				if exclude == True: tmp = tmp[~tmp.Subject.isin(exclude_subjs)]
				print('Delay = '+str(delay))
				print('Removing good/bad subjs = '+str(exclude))
				print(ps+' x '+pc)
				print(len(tmp))
				res = stats.pearsonr(tmp[ps],tmp[pc])
				result = sm2.ols(formula=pc+'~Age +'+ps,data=tmp).fit()
				print(res)
				print(result.summary()) 
				tmpplt = tmp.rename(columns={ps: ps+' Selection Rate'})
				tmpplt.rename(columns={pc: 'PC '+pc}, inplace=True)
				fig, ax = plt.subplots(1, figsize=(5, 5))
				sns.regplot(data=tmpplt,x=ps+' Selection Rate', y='PC '+pc,ax=ax)
				ax.set_title('Delay = '+str(delay)+'\n'+'Removing good/bad subjs = '+str(exclude))
				ax.text(0.05, 0.8, f'r = {res[0]:.2f}\np = {res[1]:.3f}', transform=ax.transAxes)
				
				# Partial regression plots controlling for age:
				PCreg = LinearRegression().fit(tmp[['Age']], tmp[pc])
				PSreg = LinearRegression().fit(tmp[['Age']], tmp[ps])
				PCresiduals = tmp[pc] - PCreg.predict(tmp[['Age']])
				PSresiduals = tmp[ps] - PSreg.predict(tmp[['Age']])
				residualsdf = pd.DataFrame({ps+' Selection Rate': PSresiduals, 'PC '+pc: PCresiduals})
				# Partial regression plot
				f, ax = plt.subplots(figsize=(6, 6))
				g1 = sns.regplot(data=residualsdf,x=ps+' Selection Rate', y='PC '+pc ,ax=ax)
				ax.set_title('Delay = '+str(delay)+'\n'+'Removing good/bad subjs = '+str(exclude)+'\n Controlling for Age')
				# r-value from t-value:
				t_value = result.tvalues[ps]
				df = result.df_resid
				r_value = np.sqrt(np.square(t_value) / (np.square(t_value) + df))
				ax.text(0.05, 0.8, f'r = {r_value:.2f}\np = {result.pvalues[ps]:.3f}', transform=ax.transAxes)
				
				


		