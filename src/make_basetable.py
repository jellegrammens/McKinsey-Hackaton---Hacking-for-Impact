import os, sys
import numpy as np
import pandas as pd


def load_in_training_data():
	'''
	Function that loads in data
	'''
	
	df1 = pd.read_excel('../data/Hacking for impact_data.xlsx', sheet_name = 'Patient_Care_Situation')
	df2 = pd.read_excel('../data/Hacking for impact_data.xlsx', sheet_name = 'Patient_Demographics')
	df3 = pd.read_excel('../data/Hacking for impact_data.xlsx', sheet_name = 'Patient_Previous_Conditions')

	return df1, df2, df3

def make_basetable(df1, df2, df3):

	################################
	# DF1 : Patient_Care_Situation #
	################################
    
    #remove care situation variable because it's meaningless (it has no time indiciation)
    df1.drop('ID_Patient_Care_Situation', 1, inplace=True)
    
    #remove trailing and leading blanks from string column
    df1['Treated_with_drugs'] = df1['Treated_with_drugs'].str.strip()

    #put na to empty string
    df1['Treated_with_drugs'] = df1['Treated_with_drugs'].replace(np.nan, '', regex=True)

    #split each drug into distinct columns and make dummies for each drug
    drugs = df1['Treated_with_drugs'].str.split(' ',expand=True).stack().str.get_dummies().sum(level=0)

    #concat horizontally
    df1 = pd.concat([df1, drugs], axis = 1)

    #remove original col
    df1.drop('Treated_with_drugs', 1, inplace=True)
    
    #make feature: number of drugs taken
    df1['Number_of_drugs_taken'] = df1[['DX1', 'DX2', 'DX3', 'DX4', 'DX5', 'DX6']].sum(axis = 1)
    
    
    ##############################
	# DF2 : Patient_Demographics #
	##############################
    
    #make smoke dictionary to map string values into boolean (missing value if 'Cannot say')
    smoke_dct = {'YES': 1, 'NO': 0, 'Cannot say': np.nan}
    df2['Patient_Smoker'] = df2['Patient_Smoker'].map(smoke_dct)
    
    
    #make dummy variables for rural/urban column
    rural_urban = df2.Patient_Rural_Urban.str.get_dummies().sum(level=0)
    df2 = pd.concat([df2, rural_urban], axis = 1)

    #remove original col
    df2.drop('Patient_Rural_Urban', 1, inplace=True)
    
    #remove mental stability for now because it's all t
    df2.drop('Patient_mental_condition', 1, inplace=True)
    
    #rename column for later to easily merge on same key as df1 (because df1 has all patient ids, mising data in df2/df3)
    df2.rename(columns = {'Patient_ID' : 'ID_Patient'}, inplace = True)
    
    ##############################
	# DF3 : Patient_Demographics #
	##############################

    
    #add dummies for previous condition 
    prevcond = df3.Previous_Condition.str.get_dummies().sum(level=0)
    df3 = pd.concat([df3, prevcond], 1)

    #remove original variable
    df3.drop('Previous_Condition', 1, inplace=True)
    
    #groupby and sum because some patients have multiple conditions present (more than 1 row per patient is possible)
    df3 = df3.groupby('Patient_ID').sum().reset_index()
    
    #add feature of how many prev conditions he had
    df3['Number_of_prev_cond'] = df3[['A', 'B', 'C', 'D', 'E', 'F', 'Z']].sum(axis = 1)
    
    #rename column for later to easily merge on same key as df1 (because df1 has all patient ids, mising data in df2/df3)
    df3.rename(columns = {'Patient_ID' : 'ID_Patient'}, inplace = True)
    
 
    ########################
	# FINAL_DF : Basetable #
	########################
    
    #left join because df1 has the predicted y-value, so pointless to do outer join
    df = df1.merge(df2, how = 'left', on = 'ID_Patient')
    df = df.merge(df3, how = 'left', on = 'ID_Patient')
    
    
    #if no previous conditions, put values to zero (missing values)
    df[['A', 'B', 'C', 'D', 'E', 'F', 'Z', 'Number_of_prev_cond']] = df[['A', 'B', 'C', 'D', 'E', 'F', 'Z', 'Number_of_prev_cond']].fillna(0)

    
    #reset index
    df = df.reset_index(drop = True)
    
    return df
    
    
    
    