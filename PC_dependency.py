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

year = 2

figurepath = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/Figures/'
datadf = pd.read_csv('csvs/datadf.csv')
Contdict = dd.io.load('csvs/PSPC_cont_tables.h5')
datadf = datadf[datadf['Year'] == year]


PCcols = ['Ab', 'Bc', 'Ba', 'Cb', 'Ac', 'Ca']
pair_array = [['Ab','Ac'],['Ba','Bc'],['Ca','Cb'],['Ba','Ca'],['Ac','Bc'],['Ab','Cb']]
ABC = {'A':'Place','B':'Animal','C':'Object'}

pairaccuracy = []
pairaccuracycol = []
output = [] # data, and independent model are seperate, seperate data for all pairs in pair_array
meanoutput = [] # average across pairs for data and independent model
dependencylist = [] # dependency values per subject
for subject,res_tmp in tqdm.tqdm(Contdict[year].items()):
	if np.count_nonzero(~np.isnan(res_tmp[PCcols].astype(float))) < 108:
		print(subject+' Did not complete both blocks of experiment')
	else:
		tmp = datadf[datadf['Subject']==subject]
		dep_all = np.nan*np.zeros((len(pair_array),3))
		dep_all1 = np.nan*np.zeros((len(pair_array),3))
		dep_all2 = np.nan*np.zeros((len(pair_array),3))
		age = tmp['Age'].values[0]
		accuracy = res_tmp[PCcols].stack().mean()
		accuracyfirst = res_tmp[PCcols][:9].stack().mean()
		accuracysecond =  res_tmp[PCcols][9:].stack().mean()
		delay = tmp['Delay'].values[0]
		same_day = tmp['Same Day'].values[0]
		d_ = {'Subject': subject,'Delay':delay,'Same Day':same_day}		
		for pair in PCcols:
			pairaccuracy.append(
			{
				'Subject': subject,
				'Delay':delay,
				'Same Day':same_day,
				'Age':age,
				'Pair': ABC[pair[0]]+'->'+ABC[pair[1].upper()],
				'Cue': ABC[pair[0]],
				'To-be-retrieved': ABC[pair[1].upper()],
				'Accuracy': res_tmp[pair].mean() 
			})
			d_[ABC[pair[0]]+'->'+ABC[pair[1].upper()]+' Accuracy'] =  res_tmp[pair].mean()
		pairaccuracycol.append(d_)
		for i,pair in enumerate(pair_array):
			res = res_tmp[PCcols] # no idea why this needs to be in loop
			nitems = np.count_nonzero(~np.isnan(res.astype(float))) # should be 108 (if did both blocks)
			dep = dependency (res, pair)
			resfirst = res_tmp[PCcols][:9].reset_index(drop=True)
			ressecond = res_tmp[PCcols][9:].reset_index(drop=True)
			depfirst = dependency (resfirst, pair)
			depsecond = dependency (ressecond, pair)
			if pair[0][0] == pair[1][0]:
				cue = pair[0][0]
			elif pair[0][1] == pair[1][1]:
				cue = pair[0][1].upper()
			output.append(
			{
				'Subject': subject,
				'Delay':delay,
				'Same Day':same_day,
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
			dep_all1[i]=depfirst
			dep_all2[i]=depsecond
		depmean = np.nanmean(dep_all,axis=0)
		depmean1 = np.nanmean(dep_all1,axis=0)
		depmean2 = np.nanmean(dep_all2,axis=0)
		meanoutput.append(
			{
				'Subject': subject,
				'Delay':delay,
				'Same Day':same_day,
				'Data Type':'Data',
				'Proportion of Joint Retrieval': depmean[0],
				'Age': age
			})
		meanoutput.append(
			{
				'Subject': subject,
				'Delay':delay,
				'Same Day':same_day,
				'Data Type':'Independent Model',
				'Proportion of Joint Retrieval': depmean[1],
				'Age': age
			})
		dependencylist.append(
			{
				'Subject': subject,
				'Delay':delay,
				'Same Day':same_day,
				'Same Day':tmp['Same Day'].iloc[0],
				'Dependency':depmean[0] - depmean[1],
				'Age': age,
				'Accuracy':accuracy,
				'Accuracy First':accuracyfirst,
				'Accuracy Second':accuracysecond,
				'Dependency First':depmean1[0] - depmean1[1],
				'Dependency Second':depmean2[0] - depmean2[1],
			})


		
outputdf = pd.DataFrame(output)
pairaccuracydf = pd.DataFrame(pairaccuracy)
pairaccuracycoldf = pd.DataFrame(pairaccuracycol)
meanoutputdf = pd.DataFrame(meanoutput)
dependencydf = pd.DataFrame(dependencylist)

dependencydf.to_csv('csvs/Dependency_Year_'+str(year)+'.csv',index=False)
pairaccuracydf.to_csv('csvs/PC_pairs_'+str(year)+'.csv',index=False)
pairaccuracycoldf.to_csv('csvs/PC_pairs_col_'+str(year)+'.csv',index=False)
outputdf.to_csv('csvs/PC_outputdf_'+str(year)+'.csv',index=False)

# condense outputdf
outputcond = []
outputplot = []
for subject in Contdict[year].keys():
	tmp = outputdf[outputdf['Subject']==subject]
	if len(tmp) > 0:
		d_ = {'Subject': subject,
			'Delay':tmp['Delay'].iloc[0],
			'Age': tmp['Age'].iloc[0]}
		d = {}
		for col in ['Dependency','Data','Independent Model','Dependent Model','Accuracy']:
			d[col] = tmp[col].mean()
			d2 = {'Model-type':col,'Dependency':d[col]}
			outputplot.append(dict(d_, **d2))
		outputcond.append(dict(d_, **d))
outputconddf = pd.DataFrame(outputcond)
outputplotdf = pd.DataFrame(outputplot)
outputconddf.to_csv('csvs/PC_outputconddf_'+str(year)+'.csv',index=False)
outputplotdf.to_csv('csvs/PC_outputplotdf_'+str(year)+'.csv',index=False)

# exclude all subjects with over 95% accuracy:
dependencydf = dependencydf[dependencydf['Accuracy']<0.95]

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
	sns.boxplot(data=tempdf, x="Age", y="Accuracy", hue='Age',palette="vlag",order=ordertmp, showfliers = False,legend=False)
	sns.stripplot(data=tempdf, x="Age", y="Accuracy", dodge=True,color=".3",order=ordertmp,s=10,jitter=False)
	delaydf = tempdf[tempdf.Delay]
	for subj in delaydf.Subject:
		tmp = delaydf[delaydf.Subject==subj]
		age = tmp.Age.iloc[0]
		ii = [ii for ii,v in enumerate(ordertmp) if v==age][0]
		v = ax.collections[ii].get_offsets().data[0][0]
		ax.scatter(v,tmp.Accuracy.iloc[0],marker='*',color='r',alpha=0.75, s=600,zorder=3)
	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
	plt.title(pairstring)
	plt.axhline(y=0.25, color='b', linestyle='--')
	ax.set(ylim=(0,1.05))
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
