#!/usr/bin/env python3

import tqdm as tqdm
import deepdish as dd
import pandas as pd
from dependency import dependency
import numpy as np
import math

figurepath = onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
datadf = pd.read_csv('datadf.csv')
Contdict = dd.io.load('PSPC_cont_tables.h5')
PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]

output = [] # data, and independent model are seperate, seperate data for all pairs in pair_array
meanoutput = [] # average across pairs for data and independent model
dependencylist = [] # dependency values per subject
for subject,res_tmp in tqdm.tqdm(Contdict.items()):
	dep_all = np.nan*np.zeros((len(pair_array),3))
	age = datadf['Age'][datadf['Subject']==subject].values[0]
	accuracy = res_tmp[PCcols].stack().mean() 
	for i,pair in enumerate(pair_array):
		res = res_tmp[PCcols] # no idea why this needs to be in loop
		dep = dependency (res, pair)
		output.append(
		{
			'Subject': subject,
			'Pair': pair,
			'Dependency Data': dep[0],
			'Dependency Independent Model': dep[1],
			'Dependency Dependent Model': dep[2]
		})
		dep_all[i]=dep
	depmean = np.nanmean(dep_all,axis=0)
	if np.mean(depmean) != 1:
		meanoutput.append(
			{
				'Subject': subject,
				'Data Type':'Data',
				'Proportion of Joint Retrieval': depmean[0],
				'Age': age
			})
		meanoutput.append(
			{
				'Subject': subject,
				'Data Type':'Independent Model',
				'Proportion of Joint Retrieval': depmean[1],
				'Age': age
			})
		dependencylist.append(
			{
				'Subject': subject,
				'Dependency':depmean[0] - depmean[1],
				'Age': age,
				'Accuracy':accuracy
			})
	else:
		print(subject)
		
outputdf = pd.DataFrame(output)
meanoutputdf = pd.DataFrame(meanoutput)
dependencydf = pd.DataFrame(dependencylist)
meanoutputdf['Age'] = meanoutputdf['Age'].map(lambda age: math.floor(age))
dependencydf['Age'] = dependencydf['Age'].map(lambda age: math.floor(age))
def numbers_to_words (number):
    number2word = {'4': "Four-Year-Olds", '5': "Five-Year-Olds", '6': "Six-Year-Olds",
            '7': "Seven-Year-Olds", '8': "Eight-Year-Olds", '9': "Nine-Year-Olds"}
    return " ".join(map(lambda i: number2word[i], str(number)))
meanoutputdf['Age'] = meanoutputdf['Age'].map(lambda age: numbers_to_words(age))
dependencydf['Age'] = dependencydf['Age'].map(lambda age: numbers_to_words(age))

order = ["Four-Year-Olds","Five-Year-Olds","Six-Year-Olds","Seven-Year-Olds"]
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=meanoutputdf, x="Age", y="Proportion of Joint Retrieval", hue="Data Type", palette="vlag",order=order, showfliers = False)
sns.stripplot(data=meanoutputdf, x="Age", y="Proportion of Joint Retrieval", hue="Data Type", dodge=True,color=".3",order=order)
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.savefig(figurepath+'proportion of joint retrieval.png', bbox_inches='tight', dpi=100)


fig, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=dependencydf, x="Age", y="Dependency", palette="vlag",order=order, showfliers = False)
sns.stripplot(data=dependencydf, x="Age", y="Dependency", color=".3",order=order)
fig.savefig(figurepath+'dependency.png', bbox_inches='tight', dpi=100)


g = sns.lmplot(data=dependencydf, x="Accuracy", y="Dependency")
g.savefig(figurepath+'accuracy_dependency_correlation.png', bbox_inches='tight', dpi=100)

sns.set(font_scale=1.5)
g = sns.lmplot(data=dependencydf, x="Accuracy", y="Dependency",col="Age",facet_kws=dict(sharex=False, sharey=False),col_order=order)
g.savefig(figurepath+'accuracy_dependency_correlation_age.png', bbox_inches='tight', dpi=100)


from scipy import stats
import statsmodels.stats.api as sms
from decimal import Decimal
statlist = []
depanovalist = []
tmpdf = meanoutputdf.loc[meanoutputdf['Data Type']=='Data']
jointanovalist = []
for age in order:
	agedf = dependencydf[dependencydf['Age']==age]
	res = stats.pearsonr(agedf['Accuracy'],agedf['Dependency'])
	depvals = dependencydf.loc[dependencydf['Age']==age, 'Dependency']
	depanovalist.append(list(depvals))
	ttest = stats.ttest_1samp(depvals, popmean=0)
	statlist.append({'Age':age, 'Mean':np.mean(depvals),'SE':stats.sem(depvals),'95% CI':sms.DescrStatsW(depvals).tconfint_mean(),'N':len(agedf),'df':len(depvals)-1,'t-stat':ttest[0], 'p-value':ttest[1], 'Dep/Acc r':res[0], 'Dep/Acc p':res[1]})
	jointvals = tmpdf.loc[tmpdf['Age']==age,'Proportion of Joint Retrieval']
	jointanovalist.append(list(jointvals))
statdf=pd.DataFrame(statlist)

anovadf1 = len(dependencydf) - len(depanovalist)
anovadf2 = len(depanovalist) - 1

depanova = stats.f_oneway(depanovalist[0],depanovalist[1],depanovalist[2],depanovalist[3])

jointanova = stats.f_oneway(jointanovalist[0],jointanovalist[1],jointanovalist[2],jointanovalist[3])

res = stats.pearsonr(dependencydf['Accuracy'],dependencydf['Dependency'])

#res = stats.tukey_hsd(jointanovalist[0],jointanovalist[1],jointanovalist[2],jointanovalist[3])
#print(res)



	
	