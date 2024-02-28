#!/usr/bin/env python3

# On average, how long did PS and PC take?
import os
import tqdm
import glob as glob
import pandas as pd
import datetime

year = 1

onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/'
onedrive_datapath = onedrive_path+'Data/'
datadf = pd.read_csv('csvs/datadf.csv')
datadf = datadf[datadf['Year'] == year]

dflist = []
for index, row in tqdm.tqdm(datadf.iterrows()):
	subject = row.Subject
	year = str(row.Year)
	base_path = onedrive_datapath+'Year '+str(year)+'/'+subject
	MDEM_path = base_path+'/Sess*/*test*.csv'
	MDEM_path2 = base_path+'/*test*.csv'
	MDEM_path3 = base_path+'/Year*/*test*.csv'
	MDEM_path4 = base_path+'/Year*/Sess*/*test*.csv'
	files = glob.glob(MDEM_path)+glob.glob(MDEM_path2)+glob.glob(MDEM_path3)+glob.glob(MDEM_path4)
	files = [f for f in files if os.path.getsize(f) > 50000]
	for i,f in enumerate(files):
		df = pd.read_csv(f)
		PS_end = df[df['mission6.started'].notnull() == True]['mission6.started'].iloc[0]
		PC_end = df[df['mission8.started'].notnull() == True]['mission8.started'].iloc[0] - PS_end
		dflist.append({'Subject':subject,
				  'PS_duration':PS_end,
				  'PC_duration':PC_end})
df = pd.DataFrame(dflist)
PSmean = str(datetime.timedelta(seconds=df['PS_duration'].mean()))
PCmean = str(datetime.timedelta(seconds=df['PC_duration'].mean()))
PSstd = str(datetime.timedelta(seconds=df['PS_duration'].std()))
PCstd = str(datetime.timedelta(seconds=df['PC_duration'].std()))

		
		
		
