#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 00:02:02 2022

@author: yuhan
"""


import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from scipy.spatial import cKDTree
from sklearn.svm import SVR
import geopandas as gpd
import operator 
import matplotlib.pyplot as plt
from sklearn import datasets
from ReclassifyLandUse import ReC
from DamageEstimation import DepthDamage
import matplotlib.pyplot as plt
#from confusion_matrix import confusion_matrix
###############################################################################
###############################################################################

import pickle
with open('./inputs/saved_dictionary.pkl', 'rb') as f:
    GeoNeighbors = pickle.load(f)

###############################################################################
"""
0 water
1 developed open space
2 developed low
3 developed medium
4 developed high
5 barren land
6 forest/psature/scurb
7 cultivated crops
8 wetland
"""
###############################################################################
###############################################################################
###############################################################################
#fp = "/Users/yuhan/Documents/PropertyBuyout/model_Galveston/gis/Parcel_Join.shp"
fp = "./inputs/gis_parcel_Galveston/Parcels_inputs_fixed_TF2.shp"
Geodata = gpd.read_file(fp)


pth2 = r'./outputs/Safegraph_VisitCount.csv'
SafeGraph_galvagg = pd.read_csv(pth2,encoding='utf-8')
Geodata           = Geodata.merge(SafeGraph_galvagg, 
                                  left_on='CensusBlk', 
                                  right_on='poi_cbg', how='left')


Geodata.VAL19TOT  = pd.to_numeric(Geodata.VAL19TOT.replace(',', ''), 
                                    errors='coerce')
Geodata.VAL19LAND = pd.to_numeric(Geodata.VAL19LAND.replace(',', ''),
                                    errors='coerce')
Geodata.VAL19IMP  = pd.to_numeric(Geodata.VAL19IMP.replace(',', ''),
                                    errors='coerce')

Geodata['VAL19TOT']  = Geodata['VAL19TOT'].fillna(0)
Geodata['VAL19LAND'] = Geodata['VAL19LAND'].fillna(0)
Geodata['VAL19IMP']  = Geodata['VAL19IMP'].fillna(0)
Geodata['LANDUSE']   = Geodata['LANDUSE'].fillna("")


Geodata["lu01re"] = Geodata["lu01"].apply(ReC.reclassify)
Geodata["lu06re"] = Geodata["lu06"].apply(ReC.reclassify)
Geodata["lu11re"] = Geodata["lu11"].apply(ReC.reclassify)
Geodata["lu15re"] = Geodata["lu15"].apply(ReC.reclassify)
Geodata["lu20re"] = Geodata["lu20"].apply(ReC.reclassify) 
###############################################################################

Geodata[ 'landCode' ] = Geodata["LANDUSE"].apply(ReC.reassignBuildingCode)
Geodata[ 'LUCode' ]   = Geodata["landCode"].apply(ReC.reassignBuildingCode2)
#Geodata = Geodata.apply(ReC.reassignNonLU,  axis=1)

Geodata["LUCode01re"] = Geodata[ 'LUCode' ] 
Geodata["LUCode06re"] = Geodata[ 'LUCode' ] 
Geodata["LUCode11re"] = Geodata[ 'LUCode' ] 
Geodata["LUCode15re"] = Geodata[ 'LUCode' ] 
Geodata["LUCode20re"] = Geodata[ 'LUCode' ] 

Geodata = Geodata.apply(ReC.reclassify2,  axis=1)


Geodata[ ["LUCode11re", "LUCode15re", "LUCode20re"] ]
Geodata[ ["lu20re", "LUCode", "LUCode20re"] ]
###############################################################################
pth1 = r"./inputs/Harzus_tool/flBldgStructDmgFn.csv"
pth2 = r"./inputs/Harzus_tool/flBldgContDmgFn.csv"
pth3 = r"./inputs/Harzus_tool/flBldgInvDmgFn.csv"

DamageFunc1 =pd.read_csv(pth1, encoding='utf-8')
DamageFunc2 =pd.read_csv(pth2, encoding='utf-8')
DamageFunc3 =pd.read_csv(pth3, encoding='utf-8')

print ("Finish read data")
###############################################################################
###############################################################################

############Issues with group by###############################################  
pth2 = r"./outputs/Policy_Zip.csv"
Policy_Zip =pd.read_csv(pth2, encoding='utf-8')
Policy_Zip = Policy_Zip[ ["ZipCode", "policyCost", 
                          "policyCount", "CBR", 
                          "CBRStd", "elevationN"] ]
Policy_Zip["ZipCode"] = pd.to_numeric(Policy_Zip["ZipCode"], errors='coerce')
Policy_Zip = Policy_Zip.groupby(['ZipCode']).agg({
                    "policyCost"  : ['sum'], 
                    "policyCount" : ['sum'], 
                    "CBR"         : ['max'], 
                    "CBRStd"      : ['max'], 
                    "elevationN"  : ['sum'] }).reset_index()

# Policy_Zip["CBR"]        = Policy_Zip["CBR"]    / Policy_Zip["policyCount"]
# Policy_Zip["CBRStd"]     = Policy_Zip["CBRStd"] / Policy_Zip["policyCount"]
# Policy_Zip["policyCost"] = Policy_Zip["policyCost"] / Policy_Zip["policyCount"]
Policy_Zip.columns       = ['ZipCode' , 'policyCost', 'policyCount', 
                            'CBR', 'CBRStd', 'elevationN']
Geodata = Geodata.merge(Policy_Zip, 
                        left_on='ZipCode', right_on='ZipCode', how='left')

###############################################################################
from scipy.stats import truncnorm
Geodata['CBR']     = Geodata['CBR'].fillna(0.386)
Geodata['CBRStd']  = Geodata['CBRStd'].fillna(0.095)
CBR = np.random.normal(loc= Geodata['CBR'], scale= Geodata['CBRStd']).tolist()
CBRlist = []
for idx in range( len( CBR ) ) :
    cbi = CBR[idx]
    if cbi < 0:
        cbi2 = truncnorm.rvs(0, 1, loc= Geodata['CBR'][idx], 
                             scale= Geodata['CBRStd'][idx], size=1)
        #print(cbi2)
        CBRlist.append( cbi2[0] )
    elif cbi > 1:
        cbi2 = truncnorm.rvs(0, 1, size=1)
        #print(cbi2)
        CBRlist.append( cbi2[0] )
    else:
        CBRlist.append( cbi )

A1 = DamageFunc1.loc[ DamageFunc1.Description == 'one floor, no basement, Structure, A-Zone' ].reset_index().loc[0].tolist()[5:34]
V1 = DamageFunc1.loc[ DamageFunc1.Description == 'one floor, no basement, Structure, V-Zone' ].reset_index().loc[0].tolist()[5:34]
O1 = DamageFunc1.loc[ DamageFunc1.Description == 'one story, no basement, Structure' ].reset_index().loc[0].tolist()[5:34]
inundation      = list( range(-4, 25 ) )
A1inundation    = dict(zip(inundation, A1))
V1inundation    = dict(zip(inundation, V1))
O1inundation    = dict(zip(inundation, O1))
Geodata['Content_Build_R'] = CBRlist
Geodata["ACRES" ] = pd.to_numeric(Geodata.ACRES.replace(',',''), errors='coerce')
Geodata["ACRES" ] = Geodata["ACRES" ].fillna(0)
Geodata["Cate1" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate1']), axis=1)
Geodata["Cate2" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate2']), axis=1)
Geodata["Cate3" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate3']), axis=1)
Geodata["Cate4" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate4']), axis=1)
Geodata["Cate5" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate5']), axis=1)
Geodata["Cost"]             = Geodata["VAL19TOT"]
Geodata["NumStories"]       = Geodata["N_Storie"]  
Geodata["FoundationType"]   = Geodata["Foundation"]
Geodata["FirstFloorHt"]     = DepthDamage.f( Geodata["DEM"] ) 
Geodata["Area"]             = Geodata["Bldg_Area"]
Geodata["ContentCost"]      = Geodata["VAL19TOT"] * Geodata['Content_Build_R']
Geodata["BldgDamageFnID"]   = 213
Geodata["CDDF_ID"]          = 29
Geodata["YEARBUILT"]        = Geodata["Year_Built"] 
Geodata["Tract"]            = Geodata["CensusBlk"] 
Geodata["Latitude"]         = Geodata["Lati"]
Geodata["Longitude"]        = Geodata["Long"]
floodzoneids = Geodata.floodzones.astype(int).tolist()
###############################################################################
FirstFloorHt = np.round( Geodata["FirstFloorHt"].apply(lambda x: 0 if x < 0 else x ) ).astype(int).tolist()
Cate1depth = np.round( Geodata["Cate1"] ).astype(int).tolist()
Geodata["DamagePCate1"] = DepthDamage.damage(Cate1depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate1"] = Geodata["Cost"]* Geodata["DamagePCate1"] 
Cate2depth = np.round( Geodata["Cate2"] ).astype(int).tolist()
Geodata["DamagePCate2"] = DepthDamage.damage(Cate2depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate2"] = Geodata["Cost"]* Geodata["DamagePCate2"] 
Cate3depth = np.round( Geodata["Cate3"] ).astype(int).tolist()
Geodata["DamagePCate3"] = DepthDamage.damage(Cate3depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate3"] = Geodata["Cost"]* Geodata["DamagePCate3"] 
Cate4depth = np.round( Geodata["Cate4"] ).astype(int).tolist()
Geodata["DamagePCate4"] = DepthDamage.damage(Cate4depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate4"] = Geodata["Cost"]* Geodata["DamagePCate4"] 
Cate5depth = np.round( Geodata["Cate5"] ).astype(int).tolist()
Geodata["DamagePCate5"] = DepthDamage.damage(Cate5depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate5"] = Geodata["Cost"]* Geodata["DamagePCate5"] 
Geodata["Risk"] =   ( (Geodata["DamageCate5"] + Geodata["DamageCate4"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4"] + Geodata["DamageCate3"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3"] + Geodata["DamageCate2"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2"] + Geodata["DamageCate1"])*(0.1  - 0.04)   ) / 2.0
###############################################################################
def CalDevlopedNeighbors(row, GeoNeighbors, Geodata, lucode):
    if row in GeoNeighbors:
        subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                         GeoNeighbors[ row ] ) ]
        if len( subframei)  > 0:
            alen = len( subframei.loc[ subframei["LUCode11re"] == lucode] ) * 1.0 / len( subframei) 
        else:
            alen = 0.0
        return alen
    else: 
        return 0

Geodata['NN1'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,1) )
Geodata['NN2'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,2) )
Geodata['NN3'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,3) )
Geodata['NN4'] =Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,4) )
Geodata['NN5'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,5) )
#data['舒适指数']=data.apply(lambda x:get_CHB(x['平均气温'],x['平均相对湿度'],x['2M风速']),axis=1)
# def CalDevlopedNeighbors2(GeoNeighbors, Geodata, lucode):  
#     if GeoNeighbors is not None:
#         neighbors = [float(x) for x in re.split(', |,', GeoNeighbors ) if x.isnumeric() ]
#         subframei = Geodata[ Geodata['TARGET_FID'].isin( neighbors ) ]
        
#         if len(subframei) > 0:
#             lena = len( subframei.loc[ subframei["LUCode11re"] == lucode] ) * 1.0 / len(subframei)
#         else:
#             lena = 0
#         #print( lena )
#         return lena
#     else: 
#         return 0

# Geodata['NN1'] = Geodata['NEIGHBORS'].apply( lambda x: CalDevlopedNeighbors2(x, 
#                                                     Geodata, 1) )

# Geodata['NN2'] = Geodata['NEIGHBORS'].apply( lambda x: CalDevlopedNeighbors2(x, 
#                                                     Geodata, 2) )

# Geodata['NN3'] = Geodata['NEIGHBORS'].apply( lambda x: CalDevlopedNeighbors2(x, 
#                                                     Geodata, 3) )

# Geodata['NN4'] = Geodata['NEIGHBORS'].apply( lambda x: CalDevlopedNeighbors2(x, 
#                                                     Geodata, 4) )

# Geodata['NN5'] = Geodata['NEIGHBORS'].apply( lambda x: CalDevlopedNeighbors2(x, 
#                                                     Geodata, 5) )


print ( Geodata['NN1'].value_counts() )
print ( Geodata['NN2'].value_counts() )
print ( Geodata['NN3'].value_counts() )
print ( Geodata['NN4'].value_counts() )
print ( Geodata['NN5'].value_counts() )


Geodata_select = Geodata[ ['ACRES', 'PopTot', 'TotalUnit', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65', 'BelowHigh', 'BelPoverty',
                           'Minority', 'WhitePer', 'DEM', 'Cate4', 'BeachDist',
                           'HealthDist', 'ParkDist', 'SchoolDist', 'CoastDist',
                           'WetlatDist', 'NN1', 'NN2', 'NN2', 'NN4', 'NN5'] ]

""" GBDT regression """
def ChangeState(row):
    if row.lu20re'] == 1 and row.lu20re'] != row.lu11re'] :
        return 1
    elif row.lu20re'] == 2 and row.lu20re'] != row.lu11re'] :
        return 2
    elif row.lu20re'] == 3 and row.lu20re'] != row.lu11re'] :
        return 3
    elif row.lu20re'] == 4 and row.lu20re'] != row.lu11re'] :
        return 4
    elif row.lu20re'] == 5 and row.lu20re'] != row.lu11re'] :
        return 5
    else: 
        return 0
Geodata['NewGrowth'] = Geodata.apply( ChangeState , axis=1)


# X_train, X_test, y_train, y_test = train_test_split(Geodata_select, Geodata['NewGrowth'], 
#                                                     test_size=0.4, random_state=0)
# #model training started
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth= 20), n_estimators= 3000)
# ada_clf.fit(X_train, y_train)
#model training finished


predictions             = ada_clf.predict(Geodata_select)
predictions_prob        = ada_clf.predict_proba( Geodata_select )
maxls_probs             = [ max( predicti ) for predicti in predictions_prob ]
Geodata['maxprobs']     = maxls_probs
Geodata['predLU']       = predictions
def CalUrbanGrowth(row):
    if row.predLU'] == 0:
        return 1 - row.maxprobs']
    else:
        return row.maxprobs']
Geodata['predprobs'] = Geodata.apply( CalUrbanGrowth , axis=1)


cm1 = confusion_matrix( Geodata['NewGrowth'], predictions)
print( 1 - (cm1[0, 1] + cm1[1, 0] ) / (cm1[0, 0] + cm1[1, 1] ) )
#ada_clf_val = cross_val_score(ada_clf, X_train, y_train, cv=5)
#ada_clf_val.mean()
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# Mtrix = pd.crosstab( Geodata['LUCode20re'], Geodata['LUCode11re'], 
#             rownames=['LU20'], colnames=['LU11']) 
# Mtrix2 = Mtrix / Mtrix.sum()
predYrs     = 2020
deltaY      = 5
initialYrs  = 2010
Geodata["LUC"]      = Geodata['LUCode11re']
#LandGrowth = 3276
ResidentialGrowth = (np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                                         (Geodata["LUCode20re"] < 7), "PArea" ]) - 
                     np.sum( Geodata.loc[ (Geodata["LUCode11re"] > 0) & 
                                         (Geodata["LUCode11re"] < 7), "PArea" ]) )
#ResidentialGrowth = 39883108.19182694
predyears = range(initialYrs, predYrs, deltaY)
GeoNeighbors = {}
LandGrowth        = int( np.round( ResidentialGrowth / len(predyears)) )
########
def CalDevlopedNeighbors(row, GeoNeighbors, Geodata, lucode):
    if row in GeoNeighbors:
        subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                         GeoNeighbors[ row ] ) ]
        return len( subframei.loc[ subframei["LUCode11re"] == lucode] ) * 1.0 / 16.0 
    else: 
        return 0
improvedValues  = [0]*len(Geodata)
changes         = [0]*len(Geodata)
for ykth in range( len( predyears ) ):
    """" update probs from gbdt"""
    yeark             = predyears[ ykth ]
    #LandGrowth        = int( np.round( ResidentialGrowth / len(predyears)) )
    landuseProb       = [0]*len(Geodata)
    NeighbiorLU       = [0]*len(Geodata)
    landuseType       = [0]*len(Geodata)
    # Geodata_select = Geodata[ ['ACRES', 'PopTot', 'TotalUnit', 'Mobile', 'vacant',
    #                            'MedHHInc', 'TotAge65', 'BelowHigh', 'BelPoverty',
    #                            'Minority', 'WhitePer', 'DEM', 'Cate4', 'BeachDist',
    #                            'HealthDist', 'ParkDist', 'SchoolDist', 'CoastDist',
    #                            'WetlatDist', 'NN1', 'NN2', 'NN2', 'NN4', 'NN5'] ]
    # predictions_prob = ada_clf.predict_proba( Geodata_select )
    # maxls_probs = [ predicti[1] for predicti in predictions_prob ]
    # Geodata['predprobs'] = maxls_probs
    print( yeark )
    idx = 0
    for index, row in Geodata.iterrows():
        if row.LUC']   <= 5 and row.LUC'] > 0 :
            landuseProb[idx]       =  0.0
            landuseType[idx]       =  row.LUC'] 
            improvedValues[idx]    =  row.VAL19IMP']
        elif row.LUC']   != row.LUCode11re'] :
            landuseProb[idx]       =  0.0 
            landuseType[idx]       =  row.LUC'] 
            improvedValues[idx]    =  row.VAL19IMP']
        # elif row.LUC'] == 0:
        #     landuseProb[idx]       = 0.0
        #     landuseType[idx]       = row.LUCode11re']  
        #     improvedValues[idx]    = row.VAL19IMP'] 
        # elif "UW" in row.LANDUSE'] or row.LANDUSE'] == 0 :
        #     landuseProb[idx]       = 0.0 
        #     improvedValues[idx]    = row.VAL19IMP']
        #     landuseType[idx]      =  row.LUC']    
        else:
            landuseProb[idx]  = row.predprobs']
            maxlu             = row.predLU']
            # neighborLU = [1,2,3,4,5]
            # neighborP  = [
            #     0.7 * Mtrix2[ row.LUC'] ][1] + 0.3 * row.NN1'],
            #     0.7 * Mtrix2[ row.LUC'] ][2] + 0.3 * row.NN2'],
            #     0.7 * Mtrix2[ row.LUC'] ][3] + 0.3 * row.NN3'],
            #     0.7 * Mtrix2[ row.LUC'] ][4] + 0.3 * row.NN4'],
            #     0.7 * Mtrix2[ row.LUC'] ][5] + 0.3 * row.NN5'] ]
            # if max( neighborP ) > 0:
            #     maxprob    = neighborP.index( max( neighborP ) )
            #     maxlu      = neighborLU[ maxprob ]
            # else:
            #     maxlu      = 1
                #print(  "eror ", row.LUC'], " ", 
                #      row.LUCode11re'], " ", row.LUCode20re'] )
                # if row.LUCode11re'] <=5 and row.LUCode11re'] > 0:
                #     maxlu  = row.LUCode11re']
                # else:
                #     maxlu  = -1 #why??????? why not 2,3,4,5#
            #NeighbiorLU [idx] = maxlu
            impvi = np.random.normal(159961.40, 2543.65) +\
                    np.random.normal(158.84, 55.61)* row["PArea"]
            landuseType[idx]      =  maxlu
            improvedValues[idx]   =  impvi 
        idx = idx + 1    
    #break
    Geodata["LUCprob"]  = landuseProb
    Geodata["LUCtype"]  = landuseType
    Geodata["ImproVal"] = improvedValues    
    #Geodata["LUC"]      = landuse
    Geodata.sort_values("LUCprob", ascending=False, inplace = True)
    Geodata["rank"] = Geodata.reset_index().index +1
    Geodata["PAreaCum"] = Geodata["PArea"].cumsum()
    #Geodata[ ["LUCode11re", "PArea", "PAreaCum"] ]
    lengthi = Geodata["PAreaCum"].searchsorted(LandGrowth, side='right') + 1
    print( "length of PArea", lengthi )
    Geodata.loc[ Geodata["rank"] < lengthi, "LUC"] = Geodata.loc[ Geodata["rank"] < lengthi, "LUCtype"]
    
    # Geodata.loc[ Geodata["rank"] < lengthi, "NN1"] = Geodata.loc[ Geodata["rank"] < lengthi,
    #                                                 'TARGET_FID'].apply( lambda x: 
    #                                                     CalDevlopedNeighbors(x, 
    #                                                     GeoNeighbors, Geodata, 1) )
    # Geodata.loc[ Geodata["rank"] < lengthi, "NN2"] = Geodata.loc[ Geodata["rank"] < lengthi,
    #                                                 'TARGET_FID'].apply( lambda x: 
    #                                                     CalDevlopedNeighbors(x, 
    #                                                     GeoNeighbors, Geodata, 2) )
    # Geodata.loc[ Geodata["rank"] < lengthi, "NN3"] = Geodata.loc[ Geodata["rank"] < lengthi,
    #                                                 'TARGET_FID'].apply( lambda x: 
    #                                                     CalDevlopedNeighbors(x, 
    #                                                     GeoNeighbors, Geodata, 3) )
    # Geodata.loc[ Geodata["rank"] < lengthi, "NN4"] = Geodata.loc[ Geodata["rank"] < lengthi,
    #                                                 'TARGET_FID'].apply( lambda x: 
    #                                                     CalDevlopedNeighbors(x, 
    #                                                     GeoNeighbors, Geodata, 4) )
    # Geodata.loc[ Geodata["rank"] < lengthi, "NN5"] = Geodata.loc[ Geodata["rank"] < lengthi,
    #                                                 'TARGET_FID'].apply( lambda x: 
    #                                                     CalDevlopedNeighbors(x, 
    #                                                     GeoNeighbors, Geodata, 5) )
    
    Geodata.loc[ Geodata["rank"] < lengthi, "VAL19IMP"] = Geodata.loc[ Geodata["rank"] < lengthi, "ImproVal"]
    print( "Land use 2011 ", Geodata.LUCode11re.value_counts(ascending=True) )
    print( "Land use 2020 ", Geodata.LUCode20re.value_counts(ascending=True) )
    print( "Land use prediction 2020 ", Geodata.LUC.value_counts(ascending=True) )
    print( "LU2020 ", np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                               (Geodata["LUCode20re"] < 6), "PArea" ] ), 
          " LU Predict ", np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & (Geodata["LUC"] < 6), 
                                   "PArea" ]), " \ndifference ", 
          np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                              (Geodata["LUCode20re"] < 6), "PArea" ] ) -\
          np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & (Geodata["LUC"] < 6), 
                              "PArea" ])  )


Geodata.plot( )

#Geodata.to_csv(r'/Users/yuhan/Documents/PropertyBuyout/model_Galveston/output/ParcelGC_LUC_cali.csv', index=False)

# Geodata.to_file(filename= r'/Users/yuhan/Documents/PropertyBuyout/model_Galveston/output/ParcelGC_LUC_cali.shp.zip',
#            driver='ESRI Shapefile')

###############################################################################
###############################################################################
Geodata2 = Geodata.drop( columns = ['EAL_VALT', 'EAL_VALB', 'EAL_VALP', 'SOVI_VALUE', 
                         'RESL_VALUE', 'CFLD_EVNTS', 'CFLD_AFREQ', 'CFLD_EXPB', 
                         'CFLD_EXPP', 'CFLD_EXPPE', 'CFLD_EXPT', 'CFLD_HLRB', 
                         'CFLD_HLRP', 'CFLD_HLRR', 'CFLD_EALB', 
                         'CFLD_EALP', 'CFLD_EALPE', 'CFLD_EALT', 'CFLD_EALS', 
                         'CFLD_EALR', 'CFLD_RISKS', 'CFLD_RISKR', 'RFLD_EVNTS', 
                         'RFLD_AFREQ', 'RFLD_EXPB', 'RFLD_EXPP', 'RFLD_EXPPE', 
                         'RFLD_EXPA', 'RFLD_EXPT', 'RFLD_HLRB', 'RFLD_HLRP', 
                         'RFLD_HLRA', 'RFLD_HLRR', 'RFLD_EALB', 'RFLD_EALP', 
                         'RFLD_EALPE', 'RFLD_EALA', 'RFLD_EALT', 'RFLD_EALS',
                         'RFLD_EALR', 'RFLD_RISKS', 'RFLD_RISKR' ] ) 



Geodata2.to_file(filename= r'./outputs/ParcelGC_LUC_cali2.shp.zip',
           driver='ESRI Shapefile')


#data = data.to_crs(4326) 
###############################################################################
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 1, 1, 0)
Geodata["lu11_label"] = np.where( Geodata['LUCode11re'] == 1, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 1, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)

auc0 = roc_auc_score(Geodata["lu20_label"], Geodata["lu11_label"])
print('AUC: %.3f' % auc0)

auc01 = roc_auc_score(Geodata["lu11_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc01)



Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 2, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 2, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 3, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 3, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 4, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 4, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 5, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 5, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


Geodata["lu20_label"] = np.where( Geodata['LUCode20re'] == 6, 1, 0)
Geodata["lu_predict"] = np.where( Geodata['LUC'] == 6, 1, 0)

auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"])
print('AUC: %.3f' % auc)


ns_probs = [0 for _ in range(len( Geodata["lu20_label"] ))]

ns_auc = roc_auc_score(Geodata["lu20_label"], ns_probs)
lr_auc = roc_auc_score(Geodata["lu20_label"], Geodata["lu_predict"] )

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))


Geodata['Percentile Rank'] = (Geodata.LUCpr0 - Geodata.LUCpr0.min()) / ( Geodata.LUCpr0.max() - Geodata.LUCpr0.min())

#Geodata.LUCpr0.rank(pct = True)
#ls_probs = 
ns_fpr, ns_tpr, _ = roc_curve(Geodata["lu20_label"], ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Geodata["lu20_label"], Geodata["lu_predict"])
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Null model')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Model results')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

