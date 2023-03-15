#!/usr/bin/env python3

import tqdm as tqdm
import deepdish as dd
import pandas as pd
from dependency import dependency
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.stats.api as sms
from decimal import Decimal

#figurepath = onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
#datadf = pd.read_csv('datadf.csv')
#Contdict = dd.io.load('PSPC_cont_tables.h5')

# Temporary code for looking at pilot data
figurepath = onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/Pilot/'
datadf = pd.read_csv('pilotdatadf.csv')
Contdict = dd.io.load('Pilot_cont_tables.h5')

PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]
ABC = {'A':'Place','B':'Animal','C':'Object'}

pairaccuracy = []
output = [] # data, and independent model are seperate, seperate data for all pairs in pair_array
meanoutput = [] # average across pairs for data and independent model
dependencylist = [] # dependency values per subject
for subject,res_tmp in tqdm.tqdm(Contdict.items()):
	# PILOT4 only completed Day 1:
	if np.isnan(np.sum(np.array(res_tmp)[9:,6:])):
		res_tmp = res_tmp.iloc[:9]
	dep_all = np.nan*np.zeros((len(pair_array),3))
	age = datadf['Age'][datadf['Subject']==subject].values[0]
	accuracy = res_tmp[PCcols].stack().mean()
	if not math.isnan(age):
		pilot=False if 'MDEM' in subject else True
		for pair in PCcols:
			pairaccuracy.append(
			{
				'Subject': subject,
				'Pilot':pilot,
				'Age':age,
				'Pair': ABC[pair[0]]+'->'+ABC[pair[1].upper()],
				'Cue': ABC[pair[0]],
				'To-be-retrieved': ABC[pair[1].upper()],
				'Accuracy': res_tmp[pair].mean() 
			})
		for i,pair in enumerate(pair_array):
			res = res_tmp[PCcols] # no idea why this needs to be in loop
			dep = dependency (res, pair)
			if pair[0][0] == pair[1][0]:
				cue = pair[0][0]
			elif pair[0][1] == pair[1][1]:
				cue = pair[0][1].upper()
			output.append(
			{
				'Subject': subject,
				'Pilot':pilot,
				'Age':age,
				'Pair': pair[0]+' '+pair[1],
				'Cue':cue,
				'Dependency':dep[0] - dep[1],
				'Data': dep[0],
				'Independent Model': dep[1],
				'Dependent Model': dep[2],
				'Accuracy': res_tmp[pair].stack().mean() 
			})
			dep_all[i]=dep
		depmean = np.nanmean(dep_all,axis=0)
		if np.mean(depmean) != 1:
			meanoutput.append(
				{
					'Subject': subject,
					'Pilot':pilot,
					'Data Type':'Data',
					'Proportion of Joint Retrieval': depmean[0],
					'Age': age
				})
			meanoutput.append(
				{
					'Subject': subject,
					'Pilot':pilot,
					'Data Type':'Independent Model',
					'Proportion of Joint Retrieval': depmean[1],
					'Age': age
				})
			dependencylist.append(
				{
					'Subject': subject,
					'Pilot':pilot,
					'Dependency':depmean[0] - depmean[1],
					'Age': age,
					'Accuracy':accuracy
				})
		else:
			print(subject)
		
outputdf = pd.DataFrame(output)
pairaccuracydf = pd.DataFrame(pairaccuracy)
meanoutputdf = pd.DataFrame(meanoutput)
dependencydf = pd.DataFrame(dependencylist)
outputdf['Age'] = outputdf['Age'].map(lambda age: math.floor(age))
pairaccuracydf['Age'] = pairaccuracydf['Age'].map(lambda age: math.floor(age))
meanoutputdf['Age'] = meanoutputdf['Age'].map(lambda age: math.floor(age))
dependencydf['Age'] = dependencydf['Age'].map(lambda age: math.floor(age))
def numbers_to_words (number):
    number2word = {'4': "Four-Year-Olds", '5': "Five-Year-Olds", '6': "Six-Year-Olds",
            '7': "Seven-Year-Olds", '8': "Eight-Year-Olds", '9': "Nine-Year-Olds"}
    return " ".join(map(lambda i: number2word[i], str(number)))
meanoutputdf['Age'] = meanoutputdf['Age'].map(lambda age: numbers_to_words(age))
dependencydf['Age'] = dependencydf['Age'].map(lambda age: numbers_to_words(age))

order = ["Four-Year-Olds","Five-Year-Olds","Six-Year-Olds","Seven-Year-Olds"]


fig, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=meanoutputdf, x="Age", y="Proportion of Joint Retrieval", hue="Data Type", palette="vlag",order=order, showfliers = False)
sns.stripplot(data=meanoutputdf, x="Age", y="Proportion of Joint Retrieval", hue="Data Type", dodge=True,color=".3",order=order)
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#fig.savefig(figurepath+'proportion of joint retrieval.png', bbox_inches='tight', dpi=100)

sns.set()
fig, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(data=dependencydf, x="Age", y="Dependency", palette="vlag",order=order, showfliers = False)
sns.stripplot(data=dependencydf, x="Age", y="Dependency", color=".3",order=order,jitter=False)
v = ax.collections[2].get_offsets().data[0][0]
data = list(dependencydf[(dependencydf.Pilot)]['Dependency'])
ax.scatter(v*np.ones(len(data)),data,marker='*',color='r', s=600,zorder=3)
fig.savefig(figurepath+'dependency.png', bbox_inches='tight', dpi=100)


g = sns.lmplot(data=dependencydf, x="Accuracy", y="Dependency")
#g.savefig(figurepath+'accuracy_dependency_correlation.png', bbox_inches='tight', dpi=100)

sns.set(font_scale=1.5)
g = sns.lmplot(data=dependencydf, x="Accuracy", y="Dependency",col="Age",facet_kws=dict(sharex=False, sharey=False),col_order=order)
#g.savefig(figurepath+'accuracy_dependency_correlation_age.png', bbox_inches='tight', dpi=100)


# Accuracy broken down by pair-type (6 pairs):
# Run ANOVA/t-tests to see if any pair has better accuracy?
from statsmodels.stats.anova import AnovaRM
pairacclist = []
pairaccanovalist = []
ordertmp = [4,5,6,7]
sns.set(font_scale=3)
fig = plt.figure(figsize=(30,10))
for i,pair in enumerate(PCcols):
	pairstring = ABC[pair[0]]+'->'+ABC[pair[1].upper()]
	ax = plt.subplot(1,6, i+1)
	tempdf = pairaccuracydf.loc[pairaccuracydf['Pair'] == pairstring]
	accvals = tempdf['Accuracy']
	ttest = stats.ttest_1samp(accvals, popmean=0)
	pairacclist.append({'Pair':pairstring,'N':len(tempdf),'df':len(tempdf)-1,'95% CI':sms.DescrStatsW(accvals).tconfint_mean(),'t-stat':ttest[0], 'p-value':ttest[1]})
	sns.boxplot(data=tempdf, x="Age", y="Accuracy", palette="vlag",order=ordertmp, showfliers = False)
	sns.stripplot(data=tempdf, x="Age", y="Accuracy", dodge=True,color=".3",order=ordertmp,s=10,jitter=False)
	v = ax.collections[2].get_offsets().data[0][0]
	data = list(tempdf[(tempdf.Pilot)]['Accuracy'])
	ax.scatter(v*np.ones(len(data)),data,marker='*',color='r', s=600,zorder=3)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
	plt.title(pairstring)
	plt.axhline(y=0.25, color='b', linestyle='--')
	ax.set(ylim=(0.2,1.05))
	if i>0:
		ax.yaxis.set_visible(False)
fig.tight_layout()
fig.savefig(figurepath+'accuracy_six_pairs.png', bbox_inches='tight', dpi=200)
pairaccdf=pd.DataFrame(pairacclist)
# Conduct the repeated measures ANOVA
print(AnovaRM(data=pairaccuracydf, depvar='Accuracy',
              subject='Subject', within=['Pair']).fit())
import pingouin as pg
res = pg.rm_anova(dv='Accuracy',within='Pair',subject='Subject',data=pairaccuracydf)
post_hocs = pg.pairwise_tests(dv='Accuracy',within='Pair',subject='Subject',padjust='bonf',data=pairaccuracydf)
pg.sphericity(dv='Accuracy',within='Pair',subject='Subject',data=pairaccuracydf)[-1]
pg.normality(data=pairaccuracydf,dv='Accuracy',group='Pair')
# Does Cue-type impact accuracy?
res = pg.rm_anova(dv='Accuracy',within='Cue',subject='Subject',data=pairaccuracydf)
post_hocs = pg.pairwise_tests(dv='Accuracy',within='Cue',subject='Subject',padjust='bonf',data=pairaccuracydf)
# Does item-to-be retrieved impact accuracy?
res = pg.rm_anova(dv='Accuracy',within='To-be-retrieved',subject='Subject',data=pairaccuracydf)
post_hocs = pg.pairwise_tests(dv='Accuracy',within='To-be-retrieved',subject='Subject',padjust='bonf',data=pairaccuracydf)


statlist = []
depanovalist = []
tmpdf = meanoutputdf.loc[meanoutputdf['Data Type']=='Data']
jointanovalist = []
for age in order:
	agedf = dependencydf[dependencydf['Age']==age]
	res = stats.pearsonr(agedf['Accuracy'],agedf['Dependency'])
	depvals = agedf['Dependency']
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


from pingouin import partial_corr
dependencydf = pd.DataFrame(dependencylist)
partial_corr(data=dependencydf, x='Accuracy', y='Dependency', covar='Age', method='pearson')
	
	
# Does cue 'A':'Place','B':'Animal','C':'Object' make a difference in dependency?
res = pg.rm_anova(dv='Dependency',within='Cue',subject='Subject',data=outputdf)	
post_hocs = pg.pairwise_tests(dv='Dependency',within='Cue',subject='Subject',padjust='bonf',data=outputdf)