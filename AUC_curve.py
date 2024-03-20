#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:01:50 2022

@author: yuhan
"""

import os
import re
import pickle  # 保存模块
import pandas as pd
import numpy as np
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
import Insurer
from DamageEstimation import DepthDamage
import math
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Patch
#from confusion_matrix import confusion_matrix
###############################################################################
###############################################################################

import pickle
with open('./inputs/saved_dictionary.pkl', 'rb') as f:
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
fp = "./inputs/gis_parcel_Galveston/Parcels_inputs_fixed_TF6.shp"
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
pth1 = r"./inputs/Harzus_tool/flBldgStructDmgFn.csv"
pth2 = r"./inputs/Harzus_tool/flBldgContDmgFn.csv"
pth3 = r"./inputs/Harzus_tool/flBldgInvDmgFn.csv"

DamageFunc1 =pd.read_csv(pth1, encoding='utf-8')
DamageFunc2 =pd.read_csv(pth2, encoding='utf-8')
DamageFunc3 =pd.read_csv(pth3, encoding='utf-8')

print ("Finish read data")
###############################################################################
###############################################################################
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


InsCal = Insurer.Insurer()
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
with open('./outputs/ada_clf2.pickle', 'rb') as f:
    ada_clf = pickle.load(f)
    ###predictions             = ada_clf.predict(Geodata_select)


predictions             = ada_clf.predict(Geodata_select)
predictions_probs        = ada_clf.predict_proba( Geodata_select )


Geodata_select2 = Geodata_select
Geodata_select2['predictions'] = predictions
Geodata_select2['prob1'] = [probi[0] for probi in predictions_probs]
Geodata_select2['prob2'] = [probi[1] for probi in predictions_probs]
Geodata_select2['prob3'] = [probi[2] for probi in predictions_probs]
Geodata_select2['prob4'] = [probi[3] for probi in predictions_probs]
Geodata_select2['prob5'] = [probi[4] for probi in predictions_probs]
Geodata_select2['prob6'] = [probi[5] for probi in predictions_probs]


y_label = Geodata["NewGrowth1"] 


###############################################################################
results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
results6 = []

for index, row in Geodata.iterrows():
    if row["NewGrowth1"] == 0 :
        results1.append( 1.0 )
    else:
        results1.append( 0.0 )
    
    if row["NewGrowth1"] == 1 :
        results2.append( 1.0 )
    else:
        results2.append( 0.0 )
    
    if row["NewGrowth1"] == 2  :
        results3.append( 1.0 )
    else:
        results3.append( 0.0 )
    
    if row["NewGrowth1"] == 3 :
        results4.append( 1.0 )
    else:
        results4.append( 0.0 )
    
    if row["NewGrowth1"] == 4  :
        results5.append( 1.0 )
    else:
        results5.append( 0.0 )
    
    if row["NewGrowth1"] == 5  :
        results6.append( 1.0 )
    else:
        results6.append( 0.0 )
    

    
Geodata_select2["lu0acc"] = results1
Geodata_select2["lu1acc"] = results2
Geodata_select2["lu2acc"] = results3
Geodata_select2["lu3acc"] = results4
Geodata_select2["lu4acc"] = results5
Geodata_select2["lu5acc"] = results6

################################################################################

fpr0, tpr0, _ = metrics.roc_curve( Geodata_select2["lu0acc"],  Geodata_select2['prob1'])
fpr1, tpr1, _ = metrics.roc_curve( Geodata_select2["lu1acc"],  Geodata_select2['prob2'])
fpr2, tpr2, _ = metrics.roc_curve( Geodata_select2["lu2acc"],  Geodata_select2['prob3'])
fpr3, tpr3, _ = metrics.roc_curve( Geodata_select2["lu3acc"],  Geodata_select2['prob4'])
fpr4, tpr4, _ = metrics.roc_curve( Geodata_select2["lu4acc"],  Geodata_select2['prob5'])
fpr5, tpr5, _ = metrics.roc_curve( Geodata_select2["lu5acc"],  Geodata_select2['prob6'])
################################################################################
auc0 = metrics.roc_auc_score( Geodata_select2["lu0acc"],  Geodata_select2['prob1'])
auc1 = metrics.roc_auc_score( Geodata_select2["lu1acc"],  Geodata_select2['prob2'])
auc2 = metrics.roc_auc_score( Geodata_select2["lu2acc"],  Geodata_select2['prob3'])
auc3 = metrics.roc_auc_score( Geodata_select2["lu3acc"],  Geodata_select2['prob4'])
auc4 = metrics.roc_auc_score( Geodata_select2["lu4acc"],  Geodata_select2['prob5'])
auc5 = metrics.roc_auc_score( Geodata_select2["lu5acc"],  Geodata_select2['prob6'])
################################################################################
#create ROC curve
fig = plt.figure(figsize=(6,5))

plt.plot(fpr0,tpr0, linewidth= 2 )
plt.plot(fpr1,tpr1, linewidth= 2 )
plt.plot(fpr2,tpr2, linewidth= 2 )
plt.plot(fpr4,tpr4, linewidth= 2 )
plt.plot(fpr5,tpr5, linewidth= 2 )

plt.legend(["Non-urban land(AUC = 0.952)", "Residential land(AUC = 0.959)", 
            "Commercial land(AUC = 0.960)",
            "infrastructural land(AUC = 0.899)", 
            "Public land(AUC = 0.957)"], loc ="lower right", fontsize=12)
plt.ylabel('True Positive Rate', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=14)
fig.savefig('/Users/yuhan/Desktop/ROCplot.jpg', bbox_inches='tight', dpi=300)
plt.show()



################################################################################
################################################################################