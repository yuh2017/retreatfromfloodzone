#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 18:00:29 2022

@author: yuhan
"""
import os
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
###############################################################################
###############################################################################



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
fp = "./inputs/gis_parcel_Galveston/Parcels_inputs_fixed_TF5.shp"
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
def CalDevlopedNeighbors(row, GeoNeighbors, Geodata, landuseyrs):
    if row in GeoNeighbors:
        subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                         GeoNeighbors[ row ] ) ]
        if len( subframei)  > 0:
            alen = len( subframei.loc[ subframei[landuseyrs] < 6] ) * 1.0 / len( subframei) 
        else:
            alen = 0.0
        return alen
    else: 
        return 0

###############################################################################
# def CalDevlopedNeighbori(row, GeoNeighbors, Geodata, lucode, landuseyrs):
#     if row in GeoNeighbors:
#         subframei = Geodata[ Geodata['TARGET_FID'].isin( 
#                          GeoNeighbors[ row ] ) ]
#         if len( subframei)  > 0:
#             alen = len( subframei.loc[ subframei[landuseyrs] == lucode] ) * 1.0 / len( subframei) 
#         else:
#             alen = 0.0
#         return alen
#     else: 
#         return 0
###############################################################################
Geodata['NN01'] = Geodata['TARGET_FID'].apply( lambda x: 
                                            CalDevlopedNeighbors(x, 
                                            GeoNeighbors, Geodata,
                                            "LUCode01re") )
    

Geodata['NN11'] = Geodata['TARGET_FID'].apply( lambda x: 
                                            CalDevlopedNeighbors(x, 
                                            GeoNeighbors, Geodata,
                                            "LUCode11re") )    

# Geodata['NN2'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                                     CalDevlopedNeighbors(x, 
#                                                     GeoNeighbors, Geodata,2, 
#                                                     "LUCode11re") )
# Geodata['NN3'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                                     CalDevlopedNeighbors(x, 
#                                                     GeoNeighbors, Geodata,3, 
#                                                     "LUCode11re") )
# Geodata['NN4'] =Geodata['TARGET_FID'].apply( lambda x: 
#                                                     CalDevlopedNeighbors(x, 
#                                                     GeoNeighbors, Geodata,4, 
#                                                     "LUCode11re") )
# Geodata['NN5'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                                     CalDevlopedNeighbors(x, 
#                                                     GeoNeighbors, Geodata,5, 
#                                                     "LUCode11re") )
#data['舒适指数']=data.apply(lambda x:get_CHB(x['平均气温'],x['平均相对湿度'],x['2M风速']),axis=1)
# def CalDevlopedNeighbors2(GeoNeighbors, Geodata, lucode):  
#     if GeoNeighbors is not None:
#         neighbors = [float(x) for x in re.split(', |,', GeoNeighbors ) if x.isnumeric() ]
#         subframei = Geodata[ Geodata['TARGET_FID'].isin( neighbors ) ]
        
#         if len(subframei) > 0:
#             lena = len( subframei.loc[ subframei["LUCode11re"] < 6] ) * 1.0 / len(subframei)
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


# print ( Geodata['NN1'].value_counts() )
# print ( Geodata['NN2'].value_counts() )
# print ( Geodata['NN3'].value_counts() )
# print ( Geodata['NN4'].value_counts() )
# print ( Geodata['NN5'].value_counts() )




###############################################################################
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
# def ChangeState(row):
#     if row['LUCode20re'] <= 5 and row['LUCode20re'] >= 1 :
#         return 1
#     else: 
#         return 0
Geodata['NewGrowth2'] = Geodata.apply( lambda x :
                                      ChangeState( x['LUCode11re'], 
                                                  x['LUCode20re']) , axis=1)

Geodata['NewGrowth1'] = Geodata.apply( lambda x :
                                          ChangeState( x['LUCode01re'], 
                                                      x['LUCode11re']) , axis=1)
    
    
Geodata.dropna()

# Geodata['NN1'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                              CalDevlopedNeighbors(x, 
#                                             GeoNeighbors, Geodata,1, 
#                                                     "LUCode20re") )
# Geodata['NN2'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                              CalDevlopedNeighbors(x, 
#                                             GeoNeighbors, Geodata,2, 
#                                                     "LUCode20re") )
# Geodata['NN3'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                              CalDevlopedNeighbors(x, 
#                                             GeoNeighbors, Geodata,3, 
#                                                     "LUCode20re") )
# Geodata['NN4'] =Geodata['TARGET_FID'].apply( lambda x: 
#                                             CalDevlopedNeighbors(x, 
#                                             GeoNeighbors, Geodata,4, 
#                                                     "LUCode20re") )
# Geodata['NN5'] = Geodata['TARGET_FID'].apply( lambda x: 
#                                             CalDevlopedNeighbors(x, 
#                                             GeoNeighbors, Geodata,5, 
#                                                     "LUCode20re") )

Geodata_select2 = Geodata.loc[ ( Geodata['NewGrowth2'] >= 0)].reset_index()

Geodata_select21 = Geodata_select2[ ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                            'LUCode01re', 'Risk', 'NN11' ] ]

testy_select21  = Geodata_select2['NewGrowth2']

###############################################################################
    
Geodata_select1 = Geodata.loc[ ( Geodata['LUCode01re'] <9 ) &
                              ( Geodata['NewGrowth1'] >= 0)].reset_index()
Geodata_select11 = Geodata_select1[ ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                           'LUCode11re', 'Risk','NN01'] ]

testy_select11  = Geodata_select1['NewGrowth1']


###############################################################################

Geodata_select12 = Geodata.loc[( Geodata['NewGrowth1'] > 0)].reset_index()

Geodata_select112 = Geodata_select12[ ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                           'LUCode11re', 'Risk','NN01'] ]

testy_select112  = Geodata_select12['NewGrowth1']


###############################################################################

Geodata_select11.columns = ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                           'LUCode', 'Risk','NN']


X_train, _, y_train, _ = train_test_split(Geodata_select11, testy_select11, 
                                          test_size= 0.6, random_state=0)

#######

Geodata_select112.columns = ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                            'LUCode', 'Risk','NN']

X_train2, _, y_train2, _ = train_test_split(Geodata_select112, testy_select112, 
                                          test_size= 0.3, random_state=0)



#######

Geodata_select21.columns = ['ACRES', 'PopTot', 'Mobile', 'vacant',
                           'MedHHInc', 'TotAge65','Minority', 'DEM', 'Cate4',
                           'HealthDist', 'SchoolDist', 'CoastDist', 
                           'rddist', 'rddens', 'slope_demm',
                            'LUCode', 'Risk','NN']
_      , X_test, _, y_test = train_test_split(Geodata_select21, testy_select21, 
                                              test_size= 0.6, random_state=1)


###############################################################################
print ( y_test.value_counts() )
###############################################################################

# X_train, X_test, y_train, y_test = train_test_split(Geodata_select, Geodata['NewGrowth'], 
#                                                     test_size= 0.4, random_state=0
###############################################################################

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=800)
ada_clf.fit(X_train, y_train)
predictions = ada_clf.predict(X_test)
cm1 = confusion_matrix( y_test, predictions)

print( 1 - (cm1[0, 1] + cm1[1, 0] ) / (cm1[0, 0] + cm1[1, 1] ) )

###############################################################################
ada_clf_val = cross_val_score(ada_clf, X_train, y_train, cv=5)
ada_clf_val.mean()

###############################################################################


def test_AdaBoostClassifier(*data):
    '''
    测试 AdaBoostClassifier 的用法，绘制 AdaBoostClassifier 的预测性能随基础分类器数量的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=800)
    clf.fit(X_train,y_train)
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()   


def test_AdaBoostClassifier_base_classifier(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随基础分类器数量和基础分类器的类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    from sklearn.naive_bayes import GaussianNB
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(2,1,1)
    ########### 默认的个体分类器 #############
    clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=800)
    clf.fit(X_train,y_train)
    ## 绘图
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1)
    ax.set_title("AdaBoostClassifier with Decision Tree")
    ####### Gaussian Naive Bayes 个体分类器 ########
    ax=fig.add_subplot(2,1,2)
    
    clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), learning_rate=0.1, 
                       n_estimators=800, base_estimator=GaussianNB())
    
    #clf= AdaBoostClassifier(learning_rate=0.1,base_estimator=GaussianNB())
    clf.fit(X_train,y_train)
    ## 绘图
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    ax.set_ylim(0,1)
    ax.set_title("AdaBoostClassifier with Gaussian Naive Bayes")
    plt.show()


def test_AdaBoostClassifier_learning_rate(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随学习率的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    learning_rates = np.linspace(0.1, 1, 10)
    #n_estimators = [50, 100, 200, 400, 600, 800, 1000]
    n_estimators = [800]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    traing_scores=[]
    testing_scores=[]
    ki = 0
    for learning_rate in learning_rates:
        for n_estimator in n_estimators:
            clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), 
                                learning_rate= learning_rate, 
                                n_estimators= n_estimator )
            #clf = AdaBoostClassifier(learning_rate=learning_rate,n_estimators=500)
            clf.fit(X_train,y_train)
            traing_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
            print(ki)
            ki = ki + 1
    ax.plot(learning_rates,traing_scores,label="Traing Score")
    ax.plot(learning_rates,testing_scores,label="Testing Score")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    ax.set_title("")
    plt.savefig("./Ada_learning_rate.jpg", dpi= 300)
    plt.show()

#test_AdaBoostClassifier_learning_rate(X_train, X_test, y_train, y_test )



###############################################################################

learning_rates = np.linspace(0.01, 1, 25)
# learning_rates = [ 0.01, 0.02, 0.05, 0.07, 0.1, 
#                    0.15, 0.2,  0.25, 0.3, 0.35, 
#                    0.4,  0.5,  0.6, 0.7, 0.8, 0.9, 1. ]
#learning_rates = [ 0.01, 0.05, 0.25, 0.15, 0.35 ]

n_estimators = [50, 100, 200, 400, 600, 800, 1000]

learning_rates_n = np.resize(learning_rates, len(learning_rates)*len(n_estimators) )
n_estimators_n   = np.resize(n_estimators, len(learning_rates)*len(n_estimators) )
#n_estimators = [300]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
traing_scores=[]
testing_scores=[]
ki = 0
for learning_rate in learning_rates:
    for n_estimator in n_estimators:
        clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), 
                            learning_rate= learning_rate, 
                            n_estimators= n_estimator )
        #clf = AdaBoostClassifier(learning_rate=learning_rate,n_estimators=500)
        clf.fit(X_train,y_train)
        traing_scores.append(clf.score(X_train2,y_train2))
        testing_scores.append(clf.score(X_test,y_test))
        print(ki, " training ", clf.score(X_train2,y_train2), " test ", clf.score(X_test,y_test))
        ki = ki + 1

ax.plot(learning_rates_n,traing_scores,label="Training Score")
ax.plot(learning_rates_n,testing_scores,label="Testing Score")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Score")
ax.legend(loc="best")
ax.set_title("")
#plt.savefig("/Users/yuhan/Desktop/Ada_learning_rate.jpg", dpi= 300)
plt.show()


results = pd.DataFrame()
results['learning_rates']  = learning_rates_n
results['n_estimators']    = n_estimators_n
results['traing_scores']   = traing_scores
results['testing_scores']  = testing_scores


results2 = results.pivot(index='learning_rates', columns='n_estimators', values='traing_scores')
results2.to_csv(r'./traing_scores.csv', index=True)


results3 = results.pivot(index='learning_rates', columns='n_estimators', values='testing_scores')
results3.to_csv(r'./testing_scores.csv', index=True)



# learning_rates2 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
# testing_scores2 =[0.8677, 0.865, 0.865, 0.865, 0.863, 0.865, 0.864,0.864, 0.866, 0.864, 0.864, 0.863, 0.864, 0.864, 0.864]
learning_rates = np.linspace(0.01, 1, 50)

[0.99964936, 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ]
[0.87093852, 0.86981358, 0.86930224, 0.8704418 , 0.86918537,
       0.86919998, 0.86855715, 0.86915615, 0.8704564 , 0.86930224,
       0.8699889 , 0.86813347, 0.86788511, 0.87080704, 0.86991585,
       0.86715463, 0.87022265, 0.86990124, 0.87010577, 0.86931685,
       0.87006194, 0.86819191, 0.86814808, 0.86702314, 0.87063172,
       0.86975514, 0.87004734, 0.86794355, 0.86839645, 0.86779745,
       0.8672569 , 0.86827957, 0.86949217, 0.86670173, 0.86952139,
       0.86597125, 0.86699392, 0.86718385, 0.86721307, 0.86681861,
       0.86721307, 0.86762214, 0.86664329, 0.86730072, 0.86709619,
       0.86765136, 0.86778284, 0.86728612, 0.86719846, 0.86795816]



plt.plot(learning_rates,learning_rates,label="Testing Score")




###############################################################################




def test_AdaBoostClassifier_n_estimators(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随学习率的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    learning_rates = [1]
    #np.linspace(0.1, 1, 10)
    #n_estimators = [50, 100, 200, 400, 600, 800, 1000]
    n_estimators = [50, 100, 200, 400, 600, 800, 1000]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    traing_scores=[]
    testing_scores=[]
    ki = 0
    for learning_rate in learning_rates:
        for n_estimator in n_estimators:
            clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), 
                                learning_rate= learning_rate, 
                                n_estimators= n_estimator )
            #clf = AdaBoostClassifier(learning_rate=learning_rate,n_estimators=500)
            clf.fit(X_train,y_train)
            traing_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
            print(ki)
            ki = ki + 1
    ax.plot(n_estimators,traing_scores,label="Traing score")
    ax.plot(n_estimators,testing_scores,label="Testing score")
    ax.set_xlabel("number of estimators")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


def test_AdaBoostClassifier_tree_depth(*data):
    X_train,X_test,y_train,y_test=data
    learning_rates = [1]
    #np.linspace(0.1, 1, 10)
    #n_estimators = [50, 100, 200, 400, 600, 800, 1000]
    #n_estimators = [50, 100, 200, 400, 600, 800, 1000]
    max_depths   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    traing_scores=[]
    testing_scores=[]
    ki = 0
    for learning_rate in learning_rates:
        for max_depthi in max_depths:
            clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth= max_depthi), 
                                learning_rate= learning_rate, 
                                n_estimators=  800 )
            #clf = AdaBoostClassifier(learning_rate=learning_rate,n_estimators=500)
            clf.fit(X_train,y_train)
            traing_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
            print(ki)
            ki = ki + 1
    ax.plot(max_depths,traing_scores,label="Traing score")
    ax.plot(max_depths,testing_scores,label="Testing score")
    ax.set_xlabel("number of tree depth")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()


test_AdaBoostClassifier_tree_depth(X_train, X_test, y_train, y_test )



def test_AdaBoostClassifier_algorithm(*data):
    '''
    测试  AdaBoostClassifier 的预测性能随学习率和 algorithm 参数的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    algorithms=['SAMME.R','SAMME']
    fig=plt.figure()
    learning_rates=[0.05,0.1,0.5,0.9]
    for i,learning_rate in enumerate(learning_rates):
        ax=fig.add_subplot(2,2,i+1)
        for i ,algorithm in enumerate(algorithms):
#             clf = AdaBoostClassifier(learning_rate=learning_rate,
# 				algorithm=algorithm)
            clf= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), 
                                    learning_rate= learning_rate, 
                                    n_estimators=800, algorithm=algorithm )
            
            clf.fit(X_train,y_train)
            ## 绘图
            estimators_num=len(clf.estimators_)
            X=range(1,estimators_num+1)
            ax.plot(list(X),list(clf.staged_score(X_train,y_train)),
				label="%s:Traing score"%algorithms[i])
            ax.plot(list(X),list(clf.staged_score(X_test,y_test)),
				label="%s:Testing score"%algorithms[i])
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_title("learing rate:%f"%learning_rate)
    fig.suptitle("AdaBoostClassifier")
    plt.show()


#X_train, X_test, y_train, y_test = train_test_split(Geodata_select, Geodata['NewGrowth'], 
#                                                    test_size=0.25, random_state=0)
#    test_AdaBoostClassifier(X_train,X_test,y_train,y_test) # 调用 test_AdaBoostClassifier
#    test_AdaBoostClassifier_base_classifier(X_train,X_test,y_train,y_test) # 调用 test_AdaBoostClassifier_base_classifier
#    test_AdaBoostClassifier_learning_rate(X_train,X_test,y_train,y_test) # 调用 test_AdaBoostClassifier_learning_rate
test_AdaBoostClassifier_algorithm(X_train,X_test,y_train,y_test) # 调用 test_AdaBoostClassifier_algorithm
