#!/usr/bin/env python3

# For Year 1 data (although also interesting for Years 2 and 3)
# Compare similarity ratings for items (animals, objects, and scenes)
# to PS performance on those items (Averaged across subjects)

import numpy as np
import tqdm as tqdm
import pandas as pd
import glob
import os
#import statsmodels.formula.api as sm2
import statsmodels.api as sm
import statsmodels.formula.api as smf

Year = 1
onedrive_path = 'C:/Users/tuq67942/OneDrive - Temple University/Documents/'
onedrive_datapath = onedrive_path+'Data/'

xls = pd.ExcelFile('pspc_trialbytrial.xlsx')
Answer_dict = {}
Answer_dict['Day 1'] = pd.read_excel(xls,'B2_Item_status')
Answer_dict['Day 2'] = pd.read_excel(xls,'B1_Item_status')
day2df = pd.read_excel(xls,'B1_Item_status')
day2or = pd.read_excel(xls,'Block1_proc')
for test in day2df['List_type'].unique():
    if 'PS' in test:
        minior = day2or[day2or['List_type']==test]
        for i,(index, row) in enumerate(minior.iterrows()):
            images = [row.Image_1, row.Image_2, row.Image_3, row['Image 4']]
            for image in images:
                day2df.loc[(day2df['List_type'] == test) & (day2df['Test_image'] == image), 'Trial_number'] = i+1
Answer_dict['Day 2'] = day2df

datadf = pd.read_csv('csvs/datadf.csv')
simdf = pd.read_excel('csvs/Similarity rating pspc.xlsx')



scorelist = []
for index, row in tqdm.tqdm(datadf.iterrows()):
	year = row.Year
	if year == Year and row['Same Day']:
		subject = row.Subject
		base_path = onedrive_datapath+'Year '+str(year)+'/'+subject
		MDEM_path = base_path+'/Sess*/*test*.csv'
		MDEM_path2 = base_path+'/*test*.csv'
		MDEM_path3 = base_path+'/Year*/*test*.csv'
		MDEM_path4 = base_path+'/Year*/Sess*/*test*.csv'
		files = glob.glob(MDEM_path)+glob.glob(MDEM_path2)+glob.glob(MDEM_path3)+glob.glob(MDEM_path4)
		files = [f for f in files if os.path.getsize(f) > 50000]
		for f in files:
			Day = 'Day '+f.split(onedrive_datapath)[1].split('day')[1][0]
			df = pd.read_csv(f).dropna(subset=['Event']).reset_index()
			Answers = Answer_dict[Day][Answer_dict[Day]['List_type'].str.contains('PS')]
			testdf = df.dropna(subset='PS_correct')
			for index, row2 in testdf.iterrows():
				answer = row2['mouse.clicked_image'][2:-2]
				Picture_correct = row2['ans'].split('_')[2]
				Type = Answers.loc[Answers['Test_image'] == answer]['List_type'].iloc[0].split('_')[1]
				Status = Answers.loc[Answers['Test_image'] == answer]['Status'].iloc[0]
				Status_temp = 1 if Status=='target' else 0 if Status=='lure' else -1

				scorelist.append({
					'Subject' : subject,
					'Age' : row.Age,
					'Sex' : row.Gender,
					'KBIT' : row.kbit_std_v,
					'Delay' : row.Delay,
					'Category' : Type,
					'Item' : Picture_correct,
					'Selection' : Status_temp,
					'Similarity' : simdf[simdf['Item name'] == Picture_correct]['Average ratings'].iloc[0]	
				})
				
scoredf = pd.DataFrame(scorelist)

# running some kind of mixed model with similarity, age, category, and individual subject as predictors of per-event PS performance

# already ran this model:
model1 = smf.mixedlm("Selection ~ Similarity+ Age + Category + Delay", scoredf, groups=scoredf["Subject"])
result1 = model1.fit()
display(result1.summary())

for cat in ['object', 'location', 'animal']:
	tmp = scoredf[scoredf["Category"] == cat]
	model1 = smf.mixedlm("Selection ~ Similarity+ Age + Delay", tmp, groups=tmp["Subject"])
	result1 = model1.fit()
	print(cat)
	display(result1.summary())
	
for d in [True,False]:
	tmp = scoredf[scoredf["Delay"] == d]
	model1 = smf.mixedlm("Selection ~ Similarity+ Age + Category", tmp, groups=tmp["Subject"])
	result1 = model1.fit()
	print(d)
	display(result1.summary())
	
for d in [True,False]:
	for cat in ['object', 'location', 'animal']:
		tmp = scoredf[(scoredf["Category"] == cat) & (scoredf["Delay"] == d)]
		model1 = smf.mixedlm("Selection ~ Similarity+ Age", tmp, groups=tmp["Subject"])
		result1 = model1.fit()
		print(cat,d)
		display(result1.summary())
	
	
# check and see if want to run this kind of model too:
scoredf["group"] = 1 
vcf = {'Item': '0 + C(Item)', 'Subject': '0 + C(Subject)'}
model2 = sm.MixedLM.from_formula("Selection ~ Similarity + Age + Delay", groups="Category", vc_formula=vcf, re_formula="0",data=scoredf)
result2 = model2.fit(method="cg")
result2.summary()

# From Chat GPT
# Warning: Do not run - this takes many minutes to run:
# Random effects for subjects and nested items
model = smf.mixedlm("Selection ~ Similarity + Age + Delay + Category", 
                    scoredf, 
                    groups=scoredf["Subject"], 
                    re_formula="~Category + Item")
result = model.fit()
result.summary()
	

		

