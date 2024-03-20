# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:20:10 2022

@author: zan
"""
 
import os
import re
import pickle  # 保存模块
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
from sklearn import metrics

from scipy.spatial import cKDTree
from sklearn.svm import SVR
import geopandas as gpd
import operator 
import matplotlib.pyplot as plt
from sklearn import datasets
from ReclassifyLandUse import ReC
#import Insurer
from Insurer import Insurer
from DamageEstimation import DepthDamage
import matplotlib.pyplot as plt
import math
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch

import time
from pyogrio import read_dataframe
#from confusion_matrix import confusion_matrix
###############################################################################
###############################################################################

import pickle
with open('inputs/saved_dictionary.pkl', 'rb') as f:
    GeoNeighbors = pickle.load(f)


xi1     =  -0.02768069       
mu1     =  4.29130010     
beta1   =  4.12048991

xi2     =  -0.2312489   
mu2     =  1.1366731     
beta2   =  3.2639781

xi3     =  -0.2364523      
mu3     =  -0.7983871
beta3   =  2.6085800 
def floodH( returnT, mu1, xi0, beta0):
    floodheights = []
    for returnti in returnT:
        floodHeight_mu1 =  mu1 + ( (math.pow( 1/returnti, -1* xi0) - 1)*beta0  / xi0  ) 
        floodHeight_mu2 =  mu1 + ( (math.pow( -1*math.log( 1-1/returnti ) , -1*xi0) - 1)*beta0 / xi0 ) 
        floodheights.append( floodHeight_mu1 )
    #print("Model 1 ", floodHeight_mu1, " model 2 ", floodHeight_mu2)
    return floodheights

returnT = [10,25,50,100,500]

moel3_slr03 = floodH( returnT,  mu1, xi1, beta1)
moel2_slr03 = floodH( returnT,  mu2, xi2, beta2)
moel1_slr03 = floodH( returnT,  mu3, xi3, beta3)


def PerceivedRisk( income, moel3_slr03, moel2_slr03, moel1_slr03 ):
    if income <= 3:
        return moel1_slr03
    elif income > 3 and income < 8:
        return moel2_slr03
    else:
        return moel3_slr03
    

def MeanFloodDiff(row, moel2_slr03):
    if row["Cate1" ] < 90:
        a1 = row["Cate1" ] - moel2_slr03[0]
    else:
        a1 = 0
    if row["Cate2" ] < 90:
        a2 = row["Cate2" ] - moel2_slr03[1]
    else:
        a2 = 0
    if row["Cate3" ] < 90:
        a3 = row["Cate3" ] - moel2_slr03[2]
    else:
        a3 = 0
    if row["Cate4" ] < 90:
        a4 = row["Cate4" ] - moel2_slr03[3]
    else:
        a4 = 0
    if row["Cate5" ] < 90:
        a5 = row["Cate5" ] - moel2_slr03[4]
    else: 
        a5 = 0
    mean = (a1 + a2 + a3 + a4 + a5) / 5
    return mean


def DEM_firstfloor( row ):
    if row["FirstFloorHt" ] < row["DEM_new" ]:
        return row["DEM_new" ]
    else:
        return row["FirstFloorHt" ]

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
fp = "inputs/gis_parcel_Galveston/Parcels_inputs_TF6_2.shp"
#Geodata = gpd.read_file(fp)

read_start  = time.process_time()
Geodata     = read_dataframe( fp )
read_end    = time.process_time()
Geodata     = Geodata.to_crs({'init':'epsg:4326'})

pth2        = r'outputs/Safegraph_VisitCount.csv'
SafeGraph_galvagg = pd.read_csv(pth2,encoding='utf-8')
Geodata           = Geodata.merge(SafeGraph_galvagg, 
                                  left_on='CensusBlk', 
                                  right_on='poi_cbg', how='left')
print( Geodata.crs )

Geodata.VAL19TOT  = pd.to_numeric(Geodata.VAL19TOT.replace(',', ''), 
                                    errors='coerce')
Geodata.VAL19LAND = pd.to_numeric(Geodata.VAL19LAND.replace(',', ''),
                                    errors='coerce')
Geodata.VAL19IMP  = pd.to_numeric(Geodata.VAL19IMP.replace(',', ''),
                                    errors='coerce')

Geodata["geofid"] = Geodata.index.tolist()
Geodata['VAL19LAND'] = Geodata['VAL19LAND'].fillna(0)
Geodata['VAL19IMP']  = Geodata['VAL19IMP'].fillna(0)
Geodata['VAL19TOT']  = Geodata['VAL19TOT'].fillna(0)
Geodata['LANDUSE']   = Geodata['LANDUSE'].fillna("")

Geodata['VAL19IMP']  = Geodata['VAL19TOT'] - Geodata['VAL19LAND'] 
Geodata['VAL19IMP']  = Geodata['VAL19IMP'].apply( lambda x: 0 if x < 0 else x )


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
pth1 = r"inputs/Harzus_tool/flBldgStructDmgFn.csv"
pth2 = r"inputs/Harzus_tool/flBldgContDmgFn.csv"
pth3 = r"inputs/Harzus_tool/flBldgInvDmgFn.csv"

DamageFunc1 =pd.read_csv(pth1, encoding='utf-8')
DamageFunc2 =pd.read_csv(pth2, encoding='utf-8')
DamageFunc3 =pd.read_csv(pth3, encoding='utf-8')

print ("Finish read data")
###############################################################################
###############################################################################
pth2 = r"outputs/Policy_Zip.csv"
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
########################
Geodata["Cate1" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate1']), axis=1)
Geodata["Cate2" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate2']), axis=1)
Geodata["Cate3" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate3']), axis=1)
Geodata["Cate4" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate4']), axis=1)
Geodata["Cate5" ] = Geodata.apply(lambda x: DepthDamage.catedepth(x['Cate5']), axis=1)

Geodata["DEM_adjust" ] = Geodata.apply(lambda x: MeanFloodDiff(x, moel2_slr03), axis=1)

########################
Geodata["Cate1p" ] = Geodata.apply(lambda x: PerceivedRisk(x['income'], moel3_slr03[0], moel2_slr03[0], moel1_slr03[0]), axis=1)
Geodata["Cate2p" ] = Geodata.apply(lambda x: PerceivedRisk(x['income'], moel3_slr03[1], moel2_slr03[1], moel1_slr03[0]), axis=1)
Geodata["Cate3p" ] = Geodata.apply(lambda x: PerceivedRisk(x['income'], moel3_slr03[2], moel2_slr03[2], moel1_slr03[0]), axis=1)
Geodata["Cate4p" ] = Geodata.apply(lambda x: PerceivedRisk(x['income'], moel3_slr03[3], moel2_slr03[3], moel1_slr03[0]), axis=1)
Geodata["Cate5p" ] = Geodata.apply(lambda x: PerceivedRisk(x['income'], moel3_slr03[4], moel2_slr03[4], moel1_slr03[0]), axis=1)

Geodata["DEM_new" ] = Geodata["DEM" ] + Geodata["DEM_adjust" ]
Geodata["Cost"]             = Geodata["VAL19IMP"]
Geodata["NumStories"]       = Geodata["N_Storie"]  
Geodata["FoundationType"]   = Geodata["Foundation"]
Geodata["FirstFloorHt"]     = DepthDamage.f( Geodata["DEM_new"] ) 
Geodata["FirstFloorHt" ] = Geodata.apply(lambda x: DEM_firstfloor( x ), axis=1)

Geodata["Area"]             = Geodata["Bldg_Area"]
Geodata["ContentCost"]      = Geodata["VAL19TOT"] * Geodata['Content_Build_R']
Geodata["BldgDamageFnID"]   = 213
Geodata["CDDF_ID"]          = 29
Geodata["YEARBUILT"]        = Geodata["Year_Built"] 
Geodata["Tract"]            = Geodata["CensusBlk"] 
Geodata["Latitude"]         = Geodata["Lati"]
Geodata["Longitude"]        = Geodata["Long"]
floodzoneids = Geodata.floodzones.astype(int).tolist()

InsCal = Insurer()
results1 = []
for index, row in Geodata.iterrows():
    if row["VAL19IMP"] > 0:
        inscost = InsCal.insuranceCost( row['VAL19IMP'], 
                             row['ContentCost'], row['floodzones'], 
                             row['FirstFloorHt'] ,
                             row["Cate4p" ] )[1]
        results1.append( inscost )
    else:
        results1.append( 0 )


Geodata['Insurance']    = results1
FirstFloorHt            = np.round( Geodata["FirstFloorHt"] ).astype(int).tolist() 
    
######################################################################
#FirstFloorHt = np.round( Geodata["FirstFloorHt"].apply(lambda x: 0 if x < 0 else x ) ).astype(int).tolist()
Cate1depth              = np.round( Geodata["Cate1"] ).astype(int).tolist()
Geodata["DamagePCate1"] = DepthDamage.damage(Cate1depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate1"]  = Geodata["Cost"]* Geodata["DamagePCate1"] 
Cate2depth              = np.round( Geodata["Cate2"] ).astype(int).tolist()
Geodata["DamagePCate2"] = DepthDamage.damage(Cate2depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate2"]  = Geodata["Cost"]* Geodata["DamagePCate2"] 
Cate3depth              = np.round( Geodata["Cate3"] ).astype(int).tolist()
Geodata["DamagePCate3"] = DepthDamage.damage(Cate3depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate3"]  = Geodata["Cost"]* Geodata["DamagePCate3"] 
Cate4depth              = np.round( Geodata["Cate4"] ).astype(int).tolist()
Geodata["DamagePCate4"] = DepthDamage.damage(Cate4depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate4"]  = Geodata["Cost"]* Geodata["DamagePCate4"] 
Cate5depth              = np.round( Geodata["Cate5"] ).astype(int).tolist()
Geodata["DamagePCate5"] = DepthDamage.damage(Cate5depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate5"]  = Geodata["Cost"]* Geodata["DamagePCate5"] 
Geodata["Risk"]         =   ( (Geodata["DamageCate5"] + Geodata["DamageCate4"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4"] + Geodata["DamageCate3"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3"] + Geodata["DamageCate2"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2"] + Geodata["DamageCate1"])*(0.1  - 0.04)   ) / 2.0




# Geodata2 = Geodata[ ['DEM', 'DEMmhw', 'Cate1', 'Cate2', 'Cate3', 
#                      'Cate4', 'Cate5', 'floodzones', 'buildDEM'] ]
# Geodata2.to_csv(r'/Users/yuhan/Desktop/Parcels_inputs_fixed_TF5.csv', index=True)
###############################################################################
def CalDevlopedNeighbors(row, GeoNeighbors, Geodata, lucode):
    if row in GeoNeighbors:
        subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                         GeoNeighbors[ row ] ) ]
        if len( subframei)  > 0:
            alen = len( subframei.loc[ subframei["LUCode11re"] <= 5] ) * 1.0 / len( subframei) 
        else:
            alen = 0.0
        return alen
    else: 
        return 0

Geodata['NN1'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                    CalDevlopedNeighbors(x, 
                                                    GeoNeighbors, Geodata,1) )

#print ( Geodata['NN1'].value_counts() )
# print ( Geodata['NN2'].value_counts() )
# print ( Geodata['NN3'].value_counts() )
# print ( Geodata['NN4'].value_counts() )
# print ( Geodata['NN5'].value_counts() )

Geodata = Geodata.fillna(0)
Geodata['slope_demm'] = Geodata['slope_mhw']
Geodata_select = Geodata[ ['ACRES', 'PopTot', 'TotalUnit', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65', 'BelowHigh', 'BelPoverty',
                           'Minority', 'DEM', 'Cate4', 'BeachDist',
                           'HealthDist', 'ParkDist', 'SchoolDist', 'CoastDist',
                           'WetlatDist', 'rddist', 'rddens', 'slope_demm',
                           'visit_counts', 'Risk', 'NN1'] ]


""" GBDT regression """
def ChangeState( laudcode1, landcode2):
    if landcode2  == 1 and laudcode1  != 1 :
        return 1
    elif landcode2  == 2 and laudcode1  != 2 :
        return 2
    elif landcode2  == 3 and laudcode1  != 3 :
        return 3
    elif landcode2  == 4 and laudcode1  != 4 :
        return 4
    elif landcode2  == 5 and laudcode1  != 5 :
        return 5
    else: 
        return 0



Geodata['NewGrowth2'] = Geodata.apply( lambda x :
                                      ChangeState( x['LUCode11re'], 
                                                  x['LUCode20re']) , axis=1)
Geodata['NewGrowth1'] = Geodata.apply( lambda x :
                                          ChangeState( x['LUCode01re'], 
                                                      x['LUCode11re']) , axis=1)

print ( Geodata['NewGrowth1'].value_counts() )




X_train, X_test, y_train, y_test = train_test_split(Geodata_select, Geodata['NewGrowth1'], 
                                                    test_size=0.4, random_state=0)

# # #model training started
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth= 10), 
#                              learning_rate= 0.25, 
#                              n_estimators= 400)
# ada_clf.fit(X_train, y_train)

# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth= 8), n_estimators= 500)
# ada_clf.fit(X_train, y_train)
# #model training finished

# y_predict = ada_clf.predict( X_test )
# cm1 = confusion_matrix( y_test, y_predict)
# print( 1 - (cm1[0, 1] + cm1[1, 0] ) / (cm1[0, 0] + cm1[1, 1] ) )


# with open('/Users/yuhan/Documents/PropertyBuyout/Gradient_Boost_code/outputs/ada_clf2.pickle', 'wb') as f:
#       pickle.dump(ada_clf, f)

# 读取model
with open('outputs/ada_clf2.pickle', 'rb') as f:
    ada_clf = pickle.load(f)
    ###predictions             = ada_clf.predict(Geodata_select)


predictions             = ada_clf.predict(Geodata_select)
predictions_probs        = ada_clf.predict_proba( Geodata_select )


###############################################################################
################################################################################

predictions_prob2 = []
for predicti in predictions_probs :
    predictil = list(predicti)
    rank = [index for index, value in sorted(list(enumerate(predictil)), 
                                             key=lambda x:x[1])]
    predictions_prob2.append( rank[-2] )    
    

maxls_probs             = [ max( predicti ) for predicti in predictions_probs ]
maxls_index             = [ np.argmax(predictions_probs[2]) for predicti in predictions_probs ]


Geodata['maxprobs']     = maxls_probs
Geodata['sdmaxLU']      = predictions_prob2
#Geodata['probList']     = [", ".join( [str( predi ) for predi in predicts] ) for predicts in predictions_prob ]
Geodata['maxindex']     = maxls_index
#Geodata['ResidProbs']     = [ predicti[1] for predicti in predictions_prob ]
Geodata['predLU']       = predictions

def CalUrbanGrowth(row):
    if row['predLU'] == 0:
        return 1 - row['maxprobs']
    else:
        return row['maxprobs']
Geodata['predprobs'] = Geodata.apply( CalUrbanGrowth , axis=1)

def CalUrban(row):
    if row['predLU'] == 0:
        return row['sdmaxLU'] 
    else:
        return row['predLU']
Geodata['predLU'] = Geodata.apply( CalUrban , axis=1)

#ada_clf_val = cross_val_score(ada_clf, X_train, y_train, cv=5)
#ada_clf_val.mean()
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#data = data.to_crs(4326) 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
##########################Simulation Scenario 1 #########################################
###############################################################################
# Mtrix = pd.crosstab( Geodata['LUCode20re'], Geodata['LUCode11re'], 
#             rownames=['LU20'], colnames=['LU11']) 
# Mtrix2 = Mtrix / Mtrix.sum()
predYrs     = 2050
deltaY      = 10
initialYrs  = 2020
Geodata["LUC"]      = Geodata['LUCode20re']
#LandGrowth = 3276
ResidentialGrowth = (np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                                         (Geodata["LUCode20re"] < 6), "PArea" ]) - 
                     np.sum( Geodata.loc[ (Geodata["LUCode11re"] > 0) & 
                                         (Geodata["LUCode11re"] < 6), "PArea" ]) )
#ResidentialGrowth = 39883108.19182694
predyears = range(initialYrs, predYrs, deltaY)
GeoNeighbors = {}
LandGrowth        = np.round( ResidentialGrowth )
###############################################################################
improvedValues  = [0]*len(Geodata)
changes         = [0]*len(Geodata)
for ykth in range( len( predyears ) ):
    """" update probs from gbdt"""
    yeark             = predyears[ ykth ]
    #LandGrowth        = int( np.round( ResidentialGrowth / len(predyears)) )
    landuseProb       = [0]*len(Geodata)
    NeighbiorLU       = [0]*len(Geodata)
    landuseType       = [0]*len(Geodata)
    print( yeark )
    idx = 0
    for index, row in Geodata.iterrows():
        if row['LUC']   <= 5 and row['LUC'] >= 0 and row['VAL19IMP'] > 0:
            landuseProb[idx]      +=  -1.0
            landuseType[idx]       =  row['LUC'] 
            improvedValues[idx]    =  row['VAL19IMP']
        elif "UW" in row['LANDUSE'] or row['LANDUSE'] == 0 :
            landuseProb[idx]      += -1.0 
            improvedValues[idx]    = row['VAL19IMP']
            landuseType[idx]       = 0
            row['LUC']             = 0
        elif row['LUC']   != row['LUCode20re'] :
            landuseProb[idx]       =  0.0 
            landuseType[idx]       =  row['LUC'] 
            improvedValues[idx]    =  row['VAL19IMP']  
        elif (row['poicode'] == 2204 or row['poicode'] == 2082 or 
                  row['poicode'] == 2081) :
            landuseProb[idx]       =  0.0
            landuseType[idx]       =  row['LUC']  
            improvedValues[idx]    =  row['VAL19IMP'] 
        elif row['parkigcode'] > 0 :
            landuseProb[idx]       =  0.0 
            improvedValues[idx]    =  row['VAL19IMP']
            landuseType[idx]       =  row['LUC'] 
        elif row['watercode'] > 0 :
            landuseProb[idx]       =  0.0 
            improvedValues[idx]    =  row['VAL19IMP']
            landuseType[idx]       =  row['LUC']  
        else:
            landuseProb[idx]       =  row['predprobs']
            #maxlu                  =  row['predLU'] 
            if row['predLU'] == 0:
                maxlu         = 1
                print("Error !!!!!!", row['predLU'] , " ", row['predprobs'])
            else:
                maxlu         = row['predLU']
                
            if row['LUC']   <= 5 and row['LUC'] >= 1 :
                impvi = np.random.normal(159961.40, 2543.65) +\
                        np.random.normal(158.84, 55.61) * row["PArea"] 
            else:
                impvi = 0
            landuseType[idx]       =  maxlu
            improvedValues[idx]    =  impvi 
        idx = idx + 1    
    Geodata["LUCprob"]  = landuseProb
    Geodata["LUCtype"]  = landuseType
    Geodata["ImproVal"] = improvedValues    
    #Geodata["LUC"]      = landuse
    Geodata.sort_values("LUCprob", ascending=False, inplace = True)
    Geodata["rank"]     = Geodata.reset_index().index +1
    Geodata["PAreaCum"] = Geodata["PArea"].cumsum()
    lengthi             = Geodata["PAreaCum"].searchsorted(LandGrowth, side='right') + 1
    print( "length of PArea", lengthi )
    temnlu              = len( Geodata.loc[( Geodata["LUC"] > 0 ) & (Geodata["LUC"] < 6)] )
    results1 = []
    results2 = []
    for index, row in Geodata.iterrows():
        if row["rank"] < lengthi:
            if row["LUC"] != 0:
                results1.append( row["LUCtype"] )
            else:
                results1.append( 0 ) 
        else:
            results1.append( row["LUC"] )
        if row['LUC']   <= 5 and row['LUC'] >= 1 and row["ImproVal"] == 0:
            if row['LUC']   <= 5 and row['LUC'] >= 1 :
                impvi = np.random.normal(159961.40, 2543.65) +\
                        np.random.normal(158.84, 55.61) * row["PArea"] 
            else:
                impvi = 0   
            if impvi == 0:
                print( "Improvement value error1 ", row['VAL19IMP'] )
            results2.append( impvi )
        elif row['LUC']   <= 5 and row['LUC'] >= 1 and row["ImproVal"] == 0:
            impvi = np.random.normal(159961.40, 2543.65) +\
                    np.random.normal(158.84, 55.61) * row["PArea"]     
            if impvi == 0:
                print( "Improvement value error2 ", row['VAL19IMP'] )
            results2.append( impvi )
        else:
            results2.append( row["ImproVal"] )
            
#   Geodata.loc[ Geodata["rank"] < lengthi, "LUC"]      = Geodata.loc[ Geodata["rank"] < lengthi, "LUCtype"]
#   Geodata.loc[ Geodata["rank"] < lengthi, "VAL19IMP"] = Geodata.loc[ Geodata["rank"] < lengthi, "ImproVal"]
    Geodata['LUC']      = results1
    Geodata['VAL19IMP'] = results2
    
    print( "# parcels diff ", len( Geodata.loc[( Geodata["LUC"] > 0 ) & (Geodata["LUC"] < 6)] ) - temnlu )

#   print( "Land use 2020 ", Geodata.LUCode20re.value_counts(ascending=True) )
#   print( "Land use prediction 2020 ", Geodata.LUC.value_counts(ascending=True) )
#   print( "LU2020 ", np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
#                                (Geodata["LUCode20re"] < 6), "PArea" ] ), 
#           " LU Predict ", np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & (Geodata["LUC"] < 6), 
#                                    "PArea" ]), " \ndifference ", 
#           np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
#                               (Geodata["LUCode20re"] < 6), "PArea" ] ) -\
#           np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & (Geodata["LUC"] < 6), 
#                               "PArea" ])  )
###############################################################################
###############################################################################
Geodata3 = Geodata[  ['GEOID', 'NAME', 'ZIP', 'VAL19LAND', 'VAL19IMP', 'VAL19TOT', 
                      'Bldg_Area', 'Bldg_Value', 'H_ft', 'Year_Built', 
                      'PopTot', 'TotHisp', 'MedHHInc', 'TotAge65', 
                      'Unempolyme', 'BelPoverty', 'Minority', 
                      'DEM', 'DEMmhw' , 'Cate1', 'Cate2', 'Cate3',
                      'Cate4', 'Cate5', 'lu01','lu06','lu11', 'lu15', 'lu20',
                      'BeachDist', 'HealthDist', 'ParkDist', 'SchoolDist', 
                      'ShopDist', 'CoastDist', 'WetlatDist', 'Long', 'Lati',
                      'floodzones', 'buildDEM', 'PArea', 'TARGET_FID', 
                      'NEIGHBORS', 'poicode', 'poiclass', 'poiname', 
                      'watercode', 'waterclass', 'watername', 'parkigcode', 
                      'parkigclas', 'parkigname', 'rddist', 'rddens', 
                      'slope_mhw', 'Race', 'income', 'EducationL', 'unitPrice', 
                      'LandPrice', 'geometry',  'visit_counts', 'LUCode01re', 
                      'LUCode06re', 'LUCode11re', 'LUCode15re', 'LUCode20re', 
                      'CBR', 'CBRStd','NumStories','FoundationType',
                      'FirstFloorHt', 'Area','ContentCost','BldgDamageFnID', 
                      'YEARBUILT', 'Insurance', 'DamageCate1', 'DamageCate2', 
                      'DamageCate3', 'DamageCate4', 'DamageCate5', 'geofid', 
                      'Risk', 'NN1', 'predprobs', 'LUC', 'ImproVal']  ]



#Geodata3 = Geodata3.set_crs( epsg= 2278 )




Geodata3 = gpd.GeoDataFrame( Geodata3 , geometry = 'geometry', crs="EPSG:4326")


Geodata3.to_file(filename= r'outputs/ParcelGC_LUC_sce1.shp',
           driver='ESRI Shapefile')


#data = data.to_crs(4326) 

Geodata3.loc[ ( (Geodata3['LUC'] <= 5 ) & 
             ( Geodata3['LUC'] >= 1 ) & 
             ( Geodata3['VAL19IMP'] == 0 ) ), 'VAL19IMP' ]

###############################################################################
###############################################################################
###############################################################################
###############################################################################
####################################Simulation Scenario 2 2  #####################
###############################################################################
###############################################################################
###############################################################################


color_mapping = { 0: "aqua", 1: "gold", 2: "firebrick",
                  3: "pink", 4: "mediumorchid", 5: "orange", 
                  6: "darkgreen", 7 : "limegreen"}


label_mapping = { 0: "buyout land", 1: "residential land",
                  2: "commercial land", 3: "industrial land", 
                  4: "infrastructural land", 5: "public land",
                  6: "green land", 7 : "open space"}

Geodata3["Colors"] = Geodata3["LUC"].map(color_mapping)
Geodata3["LUName"] = Geodata3["LUC"].map(label_mapping)


new_df = Geodata3.to_crs(epsg= 4326)

####################################scenario 1########################################

fig, ax = plt.subplots( figsize=(10, 8) )
pmarks = []
for ctype, data in new_df.groupby('LUC'):
    # Define the color for each group using the dictionary
    color = color_mapping[ctype]
    labeli = label_mapping[ctype]
    # Plot each group using the color defined above
    data.plot(color=color, label= labeli ,
              ax=ax, linewidth=.01,
              markersize = 20, legend=True)
    pmarks.append(Patch(facecolor=color, label=labeli))


#ax.add_artist(scale2)
#ax.add_artist(ScaleBar(1, dimension="imperial-length", units="km"))
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=[*handles,*pmarks],
          title="Simulated Land Use \nScenario 1", title_fontsize='x-large', 
          loc='lower right', fontsize= 14, frameon=True)

x, y, arrow_length = 0.8, 0.7, 0.12
ax.annotate('N', xy=(x,y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='k', width=10, headwidth=30),
                va='center',ha='center', fontsize=40,
                xycoords= ax.transAxes)

ax.add_artist(
             ScaleBar( 100, dimension="si-length", 
             units="km", location="lower center", 
             length_fraction=0.2) )

#ax.set_title('', fontsize= 28)
ax.set_axis_on()
#ctx.add_basemap(ax ,
#                source='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
#                zoom=8)
#ctx.add_basemap(ax ,source= ctx.providers.Stamen.TonerLite)
plt.tight_layout()
plt.savefig('output_images/LandUse_sce1.png',dpi=300, bbox_inches='tight', pad_inches=0)

##################################scenario 2##############################################
