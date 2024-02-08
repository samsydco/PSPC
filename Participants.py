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

onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/'
onedrive_datapath = onedrive_path+'Data/'

partlog = pd.read_excel(onedrive_path+'Testing Materials/R01 Participant Log.xlsx','Sheet1')
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
df_all = partlog.merge(df.drop_duplicates(), on=['MDEM ID'],how='left', indicator=True)
missing_demo = df_all[df_all['_merge'] == 'left_only']
sess_1_subjs = df_all[df_all['Completed S1 & S2?'] != 'X']
sess_1_subjs = sess_1_subjs[sess_1_subjs['Delay'] != False] # This only matters for Subjs in delay condition! 
print(str(len(sess_1_subjs))+' subjects did not return for the second session where PSPC memory was tested.')
df_all = df_all[df_all['Completed S1 & S2?'] == 'X']
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
datadf = datadf[datadf['Same_Day'] == True]

df_all_ = df_all.merge(datadf, on=['MDEM ID'],how='left', indicator=True)
df_all_[df_all_['_merge'] == 'left_only']

missing_elliot = ['MDEM052','MDEM055','MDEM057','MDEM058']
print(str(len)+' are missing due to experimenter error (Elliot).')


