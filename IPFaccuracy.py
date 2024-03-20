#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:10:03 2023

@author: yuhan
""" 

import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import geopandas as gpd



fp0 = r'./outputs/Galveston_SimHHs.csv'
Populationx = pd.read_csv( fp0 )

########################################################################################################################
{ "white":1,  "black": 2, "aisan":3, "Others":4 }

censusrace = Populationx.groupby([ 'censusBlk' , 'race' ]).agg({
                    'education': ['count'] 
                    }).reset_index()
censusrace.columns = [ "censusBlk", "race", "racecount" ]


table1 = pd.pivot_table(censusrace, values ='racecount', index =['censusBlk'],
                         columns =['race'], aggfunc = np.sum)
table1.reset_index(inplace=True)
table1.columns = [ "censusBlk", "white", "black", "aisan", "Others" ]
########################################################################################################################

{"BelowHigh": 1, "AboveHigh": 2}
censuseducation = Populationx.groupby([ 'censusBlk' , 'education' ]).agg({
                    'education': ['count'] 
                    }).reset_index()
censuseducation.columns = [ "censusBlk", "education", "educationcount" ]


table2 = pd.pivot_table(censuseducation, values ='educationcount', 
                        index =['censusBlk'],
                         columns =['education'], aggfunc = np.sum)
table2.reset_index(inplace=True)
table2.columns = [ "censusBlk", "BelowHigh", "AboveHigh" ]
########################################################################################################################

merged_df1 = table1.merge( table2, how='left', left_on=[ "censusBlk"], right_on=["censusBlk"])



{
 "Less$10k":     1,  
 "$15k-$20k":    2, 
 "$20k-$24k":    3,
 "$25k-$30k":    4, 
 "$30k-$35k":    5, 
 "$35k-$40k":    6,
 "$40k-$50k":    7, 
 "$45k-$50k":    8, 
 "$50k-$60k":    9, 
 "$60k-$75k":    10, 
 "$75k-$100k":   11, 
 "$100k-$125k":  12, 
 "$125k-$150k":  13, 
 "$150k-$200k":  14, 
 "$200kmore":    15  }
censusIncome = Populationx.groupby([ 'censusBlk' , 'Income' ]).agg({
                    'education': ['count'] 
                    }).reset_index()
censusIncome.columns = [ "censusBlk", "Income", "Incomecount" ]


table3 = pd.pivot_table(censusIncome, values ='Incomecount', 
                        index =['censusBlk'],
                         columns =['Income'], aggfunc = np.sum)
table3.reset_index(inplace=True)
table3.columns = [ "censusBlk", "Less$10k", "$10k-$15k",
"$15k-$20k", "$20k-$24k", "$25k-$30k", 
"$30k-$35k", "$35k-$40k", "$40k-$50k", 
"$45k-$50k", "$50k-$60k", "$60k-$75k", 
"$75k-$100k", "$100k-$125k", "$125k-$150k", 
"$150k-$200k", "$200kmore" ]


########################################################################################################################

merged_df2 = merged_df1.merge( table3, how='left', left_on=[ "censusBlk"], right_on=["censusBlk"])

merged_df2 = merged_df2.fillna(0)
merged_df2.columns =[s1 +'_s' for s1 in merged_df2.columns.tolist()]

merged_df2['censusBlk_s'] = merged_df2['censusBlk_s'].apply(int).apply(str)

########################################################################################################################

fp0 = "./Galveston_model_old_results/inputs_csv/Selected_SocioResults.csv"
GalvCendata0 = pd.read_csv( fp0 )
census_sample1  = GalvCendata0[ [ "CensusBlock", 
                              "aisan", "white", "black", "Others", "PopTot",
                              "$35k-$40k", "$75k-$100k", 
                              "$50k-$60k",  "$150k-$200k", "$200kmore", 
                              "$30k-$35k",  "$10k-$15k", "$40k-$50k", 
                              "$125k-$150k", "$45k-$50k", "$100k-$125k", 
                              "$20k-$24k", "$60k-$75k", "$25k-$30k", 
                              "$15k-$20k", "Less$10k" ] ]
census_sample1['CensusBlock'] = census_sample1['CensusBlock'].apply(int).apply(str)


########################################################################################################################
fp = "./Galveston_model_old_results/inputs_csv/GalvestonCensus.csv"
census_sample0 = gpd.read_file(fp)
census_sample2 = census_sample0[ ['CensusBlk', 'Mobile', 'Multiunit', 
                                  'SpekNonEng', 'NoVehicle', 'BelowHigh', 
                                  'BelPoverty', 'Minority', 'SinglFamiy', 
                                  'SocioStat', 'HouseDisab', 'MinorStat', 
                                  'HouTrsIdx', 'SoVI', 'ZIP_CODE'] ] 

census_sample2['BelowHigh'] =  pd.to_numeric(census_sample2.BelowHigh.replace(',', ''), 
                                    errors='coerce')

census_sample2['BelPoverty'] =  pd.to_numeric(census_sample2.BelPoverty.replace(',', ''), 
                                    errors='coerce')

census_sample2['Minority'] =  pd.to_numeric(census_sample2.Minority.replace(',', ''), 
                                    errors='coerce')

census_sample2['CensusBlk'] =  pd.to_numeric(census_sample2.CensusBlk.replace(',', ''), 
                                    errors='coerce').apply(int).apply(str)

########################################################################################################################
census_sample = census_sample1.merge(census_sample2, 
                                     left_on='CensusBlock', right_on='CensusBlk')


merged_df2 = merged_df2.merge( census_sample, how='left', left_on=[ "censusBlk_s"], right_on=["CensusBlock"])

########################################################################################################################


results1 = []
for index, row in merged_df2.iterrows():
    totalrace1 = row["white_s"] + row["black_s"] + row["aisan_s"] + row["Others_s"]
    totalrace2 = row["white"] + row["black"] + row["aisan"] + row["Others"]
    if totalrace1 > 0 and totalrace2 > 0:
        a1 = row["white_s"] / totalrace1 
        a2 = row["black_s"] / totalrace1 
        a3 = row["aisan_s"] / totalrace1 
        a4 = row["Others_s"] / totalrace1 
        
        a12 = row["white"]*1.0 / totalrace2 *1.0
        a22 = row["black"]*1.0 / totalrace2 *1.0
        a32 = row["aisan"]*1.0 / totalrace2 *1.0
        a42 = row["Others"]*1.0 / totalrace2*1.0 
        
        results1.append( {"CensusBlk": row["CensusBlk"], "simulated": a1, "actual": a12} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a2, "actual": a22} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a3, "actual": a32} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a4, "actual": a42} )
        
    # else: 
    #     a1, a2, a3, a4 = 0, 0, 0, 0
    #     a12, a22, a32, a42 = 0, 0, 0, 0
    totaleduc1 = row["BelowHigh_s"] + row["AboveHigh_s"] 
    totaleduc2 = row["PopTot"]
    if totaleduc1 > 0 and totaleduc2 > 0:
        a5 = row["BelowHigh_s"]*1.0 / totaleduc1*1.0
        a52 =  row["BelowHigh"]*1.0 / totaleduc2*1.0
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a5, "actual": a52} )
    # else:
    #     a5, a52 = 0, 0
    totalincome1 = ( row["Less$10k_s"]+  row["$10k-$15k_s"]+  row["$15k-$20k_s"]+ 
                    row["$20k-$24k_s"]+  row["$25k-$30k_s"]+  row["$30k-$35k_s"]+ 
                    row["$35k-$40k_s"]+  row["$40k-$50k_s"]+  row["$45k-$50k_s"]+ 
                    row["$50k-$60k_s"]+  row["$60k-$75k_s"]+  row["$75k-$100k_s"]+ 
                    row["$100k-$125k_s"]+ row["$125k-$150k_s"]+ row["$150k-$200k_s"]+ 
                    row["$200kmore_s"] )
    
    totalincome2 = ( row["Less$10k"]+  row["$10k-$15k"]+  row["$15k-$20k"]+ 
                    row["$20k-$24k"]+  row["$25k-$30k"]+  row["$30k-$35k"]+ 
                    row["$35k-$40k"]+  row["$40k-$50k"]+  row["$45k-$50k"]+ 
                    row["$50k-$60k"]+  row["$60k-$75k"]+  row["$75k-$100k"]+ 
                    row["$100k-$125k"]+ row["$125k-$150k"]+ row["$150k-$200k"]+ 
                    row["$200kmore"] )
        
    if totalincome1 > 0 and totalincome2 > 0:
        a6 = row["Less$10k_s"] *1.0/ totalincome1
        a7 = row["$10k-$15k_s"]*1.0 / totalincome1
        a8 = row["$15k-$20k_s"]*1.0 / totalincome1
        a9 = row["$20k-$24k_s"] / totalincome1
        a10 = row["$25k-$30k_s"] / totalincome1
        a11 = row["$30k-$35k_s"] / totalincome1
        a12 = row["$35k-$40k_s"] / totalincome1
        a13 = row["$40k-$50k_s"] / totalincome1
        a14 = row["$45k-$50k_s"] / totalincome1
        a15 = row["$50k-$60k_s"] / totalincome1
        a16 = row["$60k-$75k_s"] / totalincome1
        a17 =  row["$75k-$100k_s"] / totalincome1
        a18 =  row["$100k-$125k_s"] / totalincome1
        a19 =  row["$125k-$150k_s"] / totalincome1
        a20 =  row["$150k-$200k_s"] / totalincome1
        a21 =  row["$200kmore_s"] / totalincome1
        
        
        a62 = row["Less$10k"] *1.0/ totalincome2
        a72 = row["$10k-$15k"]*1.0 / totalincome2
        a82 = row["$15k-$20k"]*1.0 / totalincome2
        a92 = row["$20k-$24k"]*1.0 / totalincome2
        a102 = row["$25k-$30k"]*1.0 / totalincome2
        a112 = row["$30k-$35k"]*1.0 / totalincome2
        a122 = row["$35k-$40k"]*1.0 / totalincome2
        a132 = row["$40k-$50k"]*1.0 / totalincome2
        a142 = row["$45k-$50k"]*1.0 / totalincome2
        a152 = row["$50k-$60k"]*1.0 / totalincome2
        a162 = row["$60k-$75k"]*1.0 / totalincome2
        a172 =  row["$75k-$100k"]*1.0 / totalincome2
        a182 =  row["$100k-$125k"]*1.0 / totalincome2
        a192 =  row["$125k-$150k"]*1.0 / totalincome2
        a202 =  row["$150k-$200k"]*1.0 / totalincome2
        a212 =  row["$200kmore_s"]*1.0 / totalincome2
        
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a6, "actual": a62} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a7, "actual": a72} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a8, "actual": a82} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a9, "actual": a92} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a10, "actual": a102} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a11, "actual": a112} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a12, "actual": a122} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a13, "actual": a132} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a14, "actual": a142} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a15, "actual": a152} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a16, "actual": a162} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a17, "actual": a172} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a18, "actual": a182} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a19, "actual": a192} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a20, "actual": a202} )
        results1.append( {"CensusBlk": row["CensusBlk"],"simulated": a21, "actual": a212} )
    # else:
    #     a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0
    #     a62, a72, a82, a92, a102, a112, a122, a132, a142, a152, a162, a172, a182, a192, a202, a212 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0
    
    
    
    


PopulatAccuracy = pd.DataFrame( results1 )
PopulatAccuracy_gb = PopulatAccuracy.groupby([ 'CensusBlk'  ]).agg({
                    'simulated': ['sum'] ,
                    'actual': ['sum'] 
                    }).reset_index()
PopulatAccuracy_gb.columns = [ 'CensusBlk', 'simulated', 'actual' ]
PopulatAccuracy_gb.to_csv(r"./PopulatAccuracy_agg.csv", index=False)

PopulatAccuracy.to_csv(r"./PopulatAccuracy.csv", index=False)

#################