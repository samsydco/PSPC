#!/usr/bin/env python3

# Get an N
# Total number of subjects who participated in study
# Total number who have PSPC data
# etc. 

import os
import glob
import numpy as np
import pandas as pd
from itertools import compress
from datetime import datetime

onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/'
onedrive_datapath = onedrive_path+'Data/'

partlog = pd.read_excel(onedrive_path+'R01 Participant Log.xlsx','Sheet1')
partlog.columns = partlog.iloc[9]
partlog = partlog.drop(partlog.columns[0],axis=1)
partlog = partlog.drop(partlog.index[0:10])
partlog = partlog.drop(partlog.index[107:])
partlog = partlog[partlog['\n1-1 Date'].notna()]
print(str(len(partlog))+' Participated in the study.')
# Drop subject MDEM131 Due to language proficiency:
partlog = partlog[partlog['MDEM ID'] != 'MDEM131']


# Download newest demographics spreadsheet from Redcap, and put in this folder
# Add age and sex demographics
demofiles = glob.glob(onedrive_datapath+'R01MarvelousMoments*')
demofile = max(demofiles, key=os.path.getctime)
df = pd.read_csv(demofile).dropna(subset = ['demo_age'])
df = df[['redcap_event_name','session_date','participant_id','demo_age','demo_child_gender','kbit_std_v']]
df = df.dropna(subset = ['session_date'])
df = df[df['redcap_event_name'].str.contains('year1')]
df['Delay'] = df.apply(lambda row: row.session_date > "2023-03-01", axis=1)
df=df.drop(['session_date','redcap_event_name'],axis=1)
df = df.rename({'demo_age': 'Age','participant_id':'MDEM ID'}, axis='columns')
# Drop subject MDEM131 Due to language proficiency:
df = df[df['MDEM ID'] != 'MDEM131']
df_all = partlog.merge(df, on=['MDEM ID'],how='left', indicator=True)
missing_demo = df_all[df_all['_merge'] == 'left_only']
sess_1_subjs = df_all[(df_all['Completed S1 & S2?'] != 'X') & (df_all['Delay'] != False)] # This only matters for Subjs in delay condition! 
print(str(len(sess_1_subjs))+' subjects did not return for the second session where PSPC memory was tested.')
df_all = df_all[~df_all['MDEM ID'].isin(sess_1_subjs['MDEM ID'])]
df_all = df_all.drop(df_all.columns[-1],axis=1)

# Subjects with PSPC data:
data = []
# ensure that data for Day 1 and Day 2 are present
# save whether Day 1 and Day 2 were both during Session 1 or not
# a list of all subjects with data
MDEM_path = onedrive_datapath+'Year 1/MDEM*/Sess*/*test*.csv'
MDEM_path2 = onedrive_datapath+'Year 1/MDEM*/*test*.csv'
MDEM_path3 = onedrive_datapath+'Year 1/MDEM*/Year*/*test*.csv'
MDEM_path4 = onedrive_datapath+'Year 1/MDEM*/Year*/Sess*/*test*.csv'
data_tmp = glob.glob(MDEM_path)+glob.glob(MDEM_path2)+glob.glob(MDEM_path3)+glob.glob(MDEM_path4)
for d in data_tmp:
	d_ = d.split('/')[-1].split('\\')[1]
	if os.path.getsize(d) > 50000 and any(char.isdigit() for char in d_):
		if sum(d_ in s for s in data_tmp) > 1:
			session_list = [s.split('/')[-1].split('\\')[2] for s in \
							list(compress(data_tmp, [d_ in s for s in data_tmp]))]
			if session_list[0]==session_list[1] or 'MDEM' in session_list[0]:
				same_day = True  
			else: 
				same_day = False
			dictionary = {'MDEM ID':d_,'Same_Day':same_day}
			if dictionary not in data:
				data.append(dictionary)
datadf = pd.DataFrame(data)
datadf = datadf[~datadf['MDEM ID'].str.contains("TEST")]

# removing 9 o.g. version subjects:
print(str(len(datadf[datadf['Same_Day'] == False]))+' subjects did original pilot experiment.')
datadf = datadf[datadf['Same_Day'] == True]

df_all_ = df_all.merge(datadf, on=['MDEM ID'],how='left', indicator=True)
df_all_ = df_all_[df_all_['_merge'] == 'both']

missing_elliot = ['MDEM052','MDEM055','MDEM057','MDEM058']
print(str(len(missing_elliot))+' are missing due to experimenter error (Elliot).')

# Age and sex breakdown:
print('N = '+str(len(df_all_)))
print('Ages ranged from: '+str(np.min(df_all_['Age']))+' - '+\
	  str(np.max(df_all_['Age']))+' years, Mean = '+\
	  str(np.round(np.mean(df_all_['Age']),2))+\
	  ' +/- '+str(np.round(np.std(df_all_['Age']),2)))
print(df_all_['demo_child_gender'].value_counts())

# Histogram of age and gender:
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_theme(style="ticks",font_scale=1.5)
df_hist = df_all_.rename(columns={'demo_child_gender':'Sex'})
df_hist['Sex'] = df_hist['Sex'].replace({1.0: 'Male', 2.0: 'Female'})
df_hist = df_hist[df_hist['Delay']==True]
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(f)
sns.histplot(df_hist,x='Age', hue="Sex",multiple="stack",palette="hls")

# determine delay between session 1 and session 2
demofiles = glob.glob(onedrive_datapath+'R01MarvelousMoments*')
demofile = max(demofiles, key=os.path.getctime)
df = pd.read_csv(demofile)
df = df[df['redcap_event_name'].str.contains("year1")]
		
delaylist = []
for s in df_all_['MDEM ID'].unique():
	if df_all_[df_all_['MDEM ID'] == s]['Delay'].iloc[0] == True:
		dates = list(df[df['participant_id'] == s]['session_date'])
		delay = (datetime.strptime(dates[1], "%Y-%m-%d") - 
				 datetime.strptime(dates[0], "%Y-%m-%d")).days
		delaylist.append({'Subject':s,
						 'Delay Days':delay})
delaydf = pd.DataFrame(delaylist)
delaydf.to_csv('csvs/Delay_Days_1.csv',index=False)		

