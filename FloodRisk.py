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
from scipy.spatial import cKDTree
from sklearn.svm import SVR
import geopandas as gpd
import operator 
from sklearn import datasets
from ReclassifyLandUse import ReC
from DamageEstimation import DepthDamage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar

import math

#########################################################################################################
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


###############################################################################
moel3_slr0 = floodH( returnT,  mu1 , xi1, beta1)
moel2_slr0 = floodH( returnT,  mu2 , xi2, beta2)
moel1_slr0 = floodH( returnT,  mu3 , xi3, beta3)
###############################################################################
moel3_slr03 = floodH( returnT,  mu1 + 0.3, xi1, beta1)
moel2_slr03 = floodH( returnT,  mu2 + 0.3, xi2, beta2)
moel1_slr03 = floodH( returnT,  mu3 + 0.3, xi3, beta3)
###############################################################################

moel3_slr05 = floodH( returnT,  mu1 + 0.5, xi1, beta1)
moel2_slr05 = floodH( returnT,  mu2 + 0.5, xi2, beta2)
moel1_slr05 = floodH( returnT,  mu3 + 0.5, xi3, beta3)
###############################################################################

moel3_slr08 = floodH( returnT,  mu1 + 0.8, xi1, beta1)
moel2_slr08 = floodH( returnT,  mu2 + 0.8, xi2, beta2)
moel1_slr08 = floodH( returnT,  mu3 + 0.8, xi3, beta3)
###############################################################################

moel3_slr12 = floodH( returnT,  mu1 + 1.2, xi1, beta1)
moel2_slr12 = floodH( returnT,  mu2 + 1.2, xi2, beta2)
moel1_slr12 = floodH( returnT,  mu3 + 1.2, xi3, beta3)
###############################################################################

moel3_slr20 = floodH( returnT,  mu1 + 2.0, xi1, beta1)
moel2_slr20 = floodH( returnT,  mu2 + 2.0, xi2, beta2)
moel1_slr20 = floodH( returnT,  mu3 + 2.0, xi3, beta3)
###############################################################################
pth1 = r"./inputs/Harzus_tool/flBldgStructDmgFn.csv"
pth2 = r"./inputs/Harzus_tool/flBldgContDmgFn.csv"
pth3 = r"./inputs/Harzus_tool/flBldgInvDmgFn.csv"

DamageFunc1 =pd.read_csv(pth1, encoding='utf-8')
DamageFunc2 =pd.read_csv(pth2, encoding='utf-8')
DamageFunc3 =pd.read_csv(pth3, encoding='utf-8')

print ("Finish read data")
###############################################################################

fp1 = "./Gradient_Boost_code/outputs/ParcelGC_LUC_sce1.shp"
fp2 = "./Gradient_Boost_code/outputs/ParcelGC_LUC_sce2.shp"
fp3 = "./Gradient_Boost_code/outputs/ParcelGC_LUC_sce3.shp"
fp4 = "./Gradient_Boost_code/outputs/ParcelGC_LUC_sce4.shp"


Geodata = gpd.read_file(fp2)


def minLV(row):
    if row['VAL19TOT']  > row['VAL19IMP'] :
        return row['VAL19TOT'] - row['VAL19IMP']
    else:
        return 0

Geodata['VAL19TOT']  = Geodata['VAL19TOT'].fillna(0)
Geodata['VAL19IMP']  = Geodata['VAL19IMP'].fillna(0)
Geodata['VAL19LAND'] = Geodata['VAL19TOT'] - Geodata['VAL19IMP']
Geodata['VAL19LAND'] = Geodata.apply( minLV , axis=1)
Geodata['VAL19LAND'] = Geodata['VAL19LAND'].fillna(0)
Geodata['Cost'] = Geodata['VAL19TOT'].apply( lambda x: 0 if x > 276000 else x )

Geodata_exist = Geodata.loc[  Geodata["LUC"] > 0 ]
print( Geodata_exist["VAL19TOT"].sum() )
###############################################################################

Geodata_buyouts = Geodata.loc[  Geodata["LUC"] == 0 ]
print( Geodata_buyouts["Cost"].sum() )
print( Geodata_buyouts["Risk"].sum() )



Geodata_buyouts.to_file(filename= r'./outputs/Parcel_buyouts_sce4.shp.zip',
            driver='ESRI Shapefile')

###############################################################################


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



Geodata["Cate1_slr0" ] , Geodata["Cate2_slr0" ], Geodata["Cate3_slr0" ],Geodata["Cate4_slr0" ], Geodata["Cate5_slr0" ] = moel2_slr0
Geodata["Cate1u_slr0" ] ,Geodata["Cate2u_slr0" ] ,Geodata["Cate3u_slr0" ] ,Geodata["Cate4u_slr0" ] ,Geodata["Cate5u_slr0" ] = moel3_slr0
Geodata["Cate1l_slr0" ] ,Geodata["Cate2l_slr0" ] ,Geodata["Cate3l_slr0" ] ,Geodata["Cate4l_slr0" ] ,Geodata["Cate5l_slr0" ] = moel1_slr0


Geodata["Cate1_slr03" ] , Geodata["Cate2_slr03" ], Geodata["Cate3_slr03" ],Geodata["Cate4_slr03" ], Geodata["Cate5_slr03" ] = moel2_slr03
Geodata["Cate1u_slr03" ] ,Geodata["Cate2u_slr03" ] ,Geodata["Cate3u_slr03" ] ,Geodata["Cate4u_slr03" ] ,Geodata["Cate5u_slr03" ] = moel3_slr03
Geodata["Cate1l_slr03" ] ,Geodata["Cate2l_slr03" ] ,Geodata["Cate3l_slr03" ] ,Geodata["Cate4l_slr03" ] ,Geodata["Cate5l_slr03" ] = moel1_slr03


Geodata["Cate1_slr05" ] , Geodata["Cate2_slr05" ], Geodata["Cate3_slr05" ],Geodata["Cate4_slr05" ], Geodata["Cate5_slr05" ] = moel2_slr05
Geodata["Cate1u_slr05" ] ,Geodata["Cate2u_slr05" ] ,Geodata["Cate3u_slr05" ] ,Geodata["Cate4u_slr05" ] ,Geodata["Cate5u_slr05" ] = moel3_slr05
Geodata["Cate1l_slr05" ] ,Geodata["Cate2l_slr05" ] ,Geodata["Cate3l_slr05" ] ,Geodata["Cate4l_slr05" ] ,Geodata["Cate5l_slr05" ] = moel1_slr05


Geodata["Cate1_slr08" ] , Geodata["Cate2_slr08" ], Geodata["Cate3_slr08" ],Geodata["Cate4_slr08" ], Geodata["Cate5_slr08" ] = moel2_slr08
Geodata["Cate1u_slr08" ] ,Geodata["Cate2u_slr08" ] ,Geodata["Cate3u_slr08" ] ,Geodata["Cate4u_slr08" ] ,Geodata["Cate5u_slr08" ] = moel3_slr08
Geodata["Cate1l_slr08" ] ,Geodata["Cate2l_slr08" ] ,Geodata["Cate3l_slr08" ] ,Geodata["Cate4l_slr08" ] ,Geodata["Cate5l_slr08" ] = moel1_slr08


Geodata["Cate1_slr12" ] , Geodata["Cate2_slr12" ], Geodata["Cate3_slr12" ],Geodata["Cate4_slr12" ], Geodata["Cate5_slr12" ] = moel2_slr12
Geodata["Cate1u_slr12" ] ,Geodata["Cate2u_slr12" ] ,Geodata["Cate3u_slr12" ] ,Geodata["Cate4u_slr12" ] ,Geodata["Cate5u_slr12" ] = moel3_slr12
Geodata["Cate1l_slr12" ] ,Geodata["Cate2l_slr12" ] ,Geodata["Cate3l_slr12" ] ,Geodata["Cate4l_slr12" ] ,Geodata["Cate5l_slr12" ] = moel1_slr12


Geodata["Cate1_slr20" ] , Geodata["Cate2_slr20" ], Geodata["Cate3_slr20" ],Geodata["Cate4_slr20" ], Geodata["Cate5_slr20" ] = moel2_slr20
Geodata["Cate1u_slr20" ] ,Geodata["Cate2u_slr20" ] ,Geodata["Cate3u_slr20" ] ,Geodata["Cate4u_slr20" ] ,Geodata["Cate5u_slr20" ] = moel3_slr20
Geodata["Cate1l_slr20" ] ,Geodata["Cate2l_slr20" ] ,Geodata["Cate3l_slr20" ] ,Geodata["Cate4l_slr20" ] ,Geodata["Cate5l_slr20" ] = moel1_slr20



Geodata["Cost"]             = Geodata["VAL19TOT"]
Geodata["FoundationType"]   = Geodata["Foundation"]
Geodata["FirstFloorHt"]     = DepthDamage.f( Geodata["DEM"] ) 
Geodata["Area"]             = Geodata["Bldg_Area"]
Geodata["ContentCost"]      = Geodata["VAL19TOT"] * Geodata['Content_Build_R']
Geodata["BldgDamageFnID"]   = 213
Geodata["CDDF_ID"]          = 29
Geodata["YEARBUILT"]        = Geodata["Year_Built"] 
Geodata["Latitude"]         = Geodata["Lati"]
Geodata["Longitude"]        = Geodata["Long"]
floodzoneids = Geodata.floodzones.astype(int).tolist()
###############################################################################
FirstFloorHt = np.round( Geodata["FirstFloorHt"].apply(lambda x: 0 if x < 0 else x ) ).astype(int).tolist()


Cate1depth = []
Cate1depth.append( np.round( Geodata["Cate1_slr0"] ).astype(int).tolist() )
Cate1depth.append( np.round( Geodata["Cate1_slr03"] ).astype(int).tolist() )
Cate1depth.append( np.round( Geodata["Cate1_slr05"] ).astype(int).tolist() )
Cate1depth.append( np.round( Geodata["Cate1_slr08"] ).astype(int).tolist() )
Cate1depth.append( np.round( Geodata["Cate1_slr12"] ).astype(int).tolist() )
Cate1depth.append( np.round( Geodata["Cate1_slr20"] ).astype(int).tolist() )



Geodata["DamagePCate1_slr0"], Geodata["DamagePCate1_slr03"],  Geodata["DamagePCate1_slr05"], Geodata["DamagePCate1_slr08"], Geodata["DamagePCate1_slr12"], Geodata["DamagePCate1_slr20"] = DepthDamage.listOfdamage(Cate1depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate1_slr0"]  = Geodata["Cost"]* Geodata["DamagePCate1_slr0"] 
Geodata["DamageCate1_slr03"] = Geodata["Cost"]* Geodata["DamagePCate1_slr03"] 
Geodata["DamageCate1_slr05"] = Geodata["Cost"]* Geodata["DamagePCate1_slr05"] 
Geodata["DamageCate1_slr08"] = Geodata["Cost"]* Geodata["DamagePCate1_slr08"] 
Geodata["DamageCate1_slr12"] = Geodata["Cost"]* Geodata["DamagePCate1_slr12"] 
Geodata["DamageCate1_slr20"] = Geodata["Cost"]* Geodata["DamagePCate1_slr20"] 


###############################################################################
Cate1depthu = []
Cate1depthu.append( np.round( Geodata["Cate1u_slr0"] ).astype(int).tolist() )
Cate1depthu.append( np.round( Geodata["Cate1u_slr03"] ).astype(int).tolist() )
Cate1depthu.append( np.round( Geodata["Cate1u_slr05"] ).astype(int).tolist() )
Cate1depthu.append( np.round( Geodata["Cate1u_slr08"] ).astype(int).tolist() )
Cate1depthu.append( np.round( Geodata["Cate1u_slr12"] ).astype(int).tolist() )
Cate1depthu.append( np.round( Geodata["Cate1u_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate1u_slr0"], Geodata["DamagePCate1u_slr03"], Geodata["DamagePCate1u_slr05"], Geodata["DamagePCate1u_slr08"], Geodata["DamagePCate1u_slr12"], Geodata["DamagePCate1u_slr20"] = DepthDamage.listOfdamage(Cate1depthu, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate1u_slr0"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr0"] 
Geodata["DamageCate1u_slr03"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr03"] 
Geodata["DamageCate1u_slr05"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr05"] 
Geodata["DamageCate1u_slr08"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr08"] 
Geodata["DamageCate1u_slr12"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr12"] 
Geodata["DamageCate1u_slr20"] = Geodata["Cost"]* Geodata["DamagePCate1u_slr20"] 
###############################################################################
Cate1depthl = []
Cate1depthl.append( np.round( Geodata["Cate1l_slr0"] ).astype(int).tolist() )

Cate1depthl.append( np.round( Geodata["Cate1l_slr03"] ).astype(int).tolist() )
Cate1depthl.append( np.round( Geodata["Cate1l_slr05"] ).astype(int).tolist() )
Cate1depthl.append( np.round( Geodata["Cate1l_slr08"] ).astype(int).tolist() )
Cate1depthl.append( np.round( Geodata["Cate1l_slr12"] ).astype(int).tolist() )
Cate1depthl.append( np.round( Geodata["Cate1l_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate1l_slr0"], Geodata["DamagePCate1l_slr03"], Geodata["DamagePCate1l_slr05"], Geodata["DamagePCate1l_slr08"], Geodata["DamagePCate1l_slr12"], Geodata["DamagePCate1l_slr20"] = DepthDamage.listOfdamage(Cate1depthl, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate1l_slr0"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr0"] 

Geodata["DamageCate1l_slr03"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr03"] 
Geodata["DamageCate1l_slr05"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr05"] 
Geodata["DamageCate1l_slr08"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr08"] 
Geodata["DamageCate1l_slr12"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr12"] 
Geodata["DamageCate1l_slr20"] = Geodata["Cost"]* Geodata["DamagePCate1l_slr20"] 

###############################################################################


Cate2depth = []
Cate2depth.append( np.round( Geodata["Cate2_slr0"] ).astype(int).tolist() )

Cate2depth.append( np.round( Geodata["Cate2_slr03"] ).astype(int).tolist() )
Cate2depth.append( np.round( Geodata["Cate2_slr05"] ).astype(int).tolist() )
Cate2depth.append( np.round( Geodata["Cate2_slr08"] ).astype(int).tolist() )
Cate2depth.append( np.round( Geodata["Cate2_slr12"] ).astype(int).tolist() )
Cate2depth.append( np.round( Geodata["Cate2_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate2_slr0"], Geodata["DamagePCate2_slr03"], Geodata["DamagePCate2_slr05"], Geodata["DamagePCate2_slr08"], Geodata["DamagePCate2_slr12"], Geodata["DamagePCate2_slr20"]  = DepthDamage.listOfdamage(Cate2depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate2_slr0"] = Geodata["Cost"]* Geodata["DamagePCate2_slr0"] 

Geodata["DamageCate2_slr03"] = Geodata["Cost"]* Geodata["DamagePCate2_slr03"] 
Geodata["DamageCate2_slr05"] = Geodata["Cost"]* Geodata["DamagePCate2_slr05"] 
Geodata["DamageCate2_slr08"] = Geodata["Cost"]* Geodata["DamagePCate2_slr08"] 
Geodata["DamageCate2_slr12"] = Geodata["Cost"]* Geodata["DamagePCate2_slr12"] 
Geodata["DamageCate2_slr20"] = Geodata["Cost"]* Geodata["DamagePCate2_slr20"] 
###############################################################################

Cate2depthl = []
Cate2depthl.append( np.round( Geodata["Cate2l_slr0"] ).astype(int).tolist() )

Cate2depthl.append( np.round( Geodata["Cate2l_slr03"] ).astype(int).tolist() )
Cate2depthl.append( np.round( Geodata["Cate2l_slr05"] ).astype(int).tolist() )
Cate2depthl.append( np.round( Geodata["Cate2l_slr08"] ).astype(int).tolist() )
Cate2depthl.append( np.round( Geodata["Cate2l_slr12"] ).astype(int).tolist() )
Cate2depthl.append( np.round( Geodata["Cate2l_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate2l_slr0"], Geodata["DamagePCate2l_slr03"], Geodata["DamagePCate2l_slr05"], Geodata["DamagePCate2l_slr08"], Geodata["DamagePCate2l_slr12"], Geodata["DamagePCate2l_slr20"] = DepthDamage.listOfdamage(Cate2depthl, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate2l_slr0"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr0"] 

Geodata["DamageCate2l_slr03"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr03"] 
Geodata["DamageCate2l_slr05"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr05"] 
Geodata["DamageCate2l_slr08"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr08"] 
Geodata["DamageCate2l_slr12"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr12"] 
Geodata["DamageCate2l_slr20"] = Geodata["Cost"]* Geodata["DamagePCate2l_slr20"] 

###############################################################################
Cate2depthu = []
Cate2depthu.append( np.round( Geodata["Cate2u_slr0"] ).astype(int).tolist() )

Cate2depthu.append( np.round( Geodata["Cate2u_slr03"] ).astype(int).tolist() )
Cate2depthu.append( np.round( Geodata["Cate2u_slr05"] ).astype(int).tolist() )
Cate2depthu.append( np.round( Geodata["Cate2u_slr08"] ).astype(int).tolist() )
Cate2depthu.append( np.round( Geodata["Cate2u_slr12"] ).astype(int).tolist() )
Cate2depthu.append( np.round( Geodata["Cate2u_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate2u_slr0"],Geodata["DamagePCate2u_slr03"], Geodata["DamagePCate2u_slr05"], Geodata["DamagePCate2u_slr08"], Geodata["DamagePCate2u_slr12"], Geodata["DamagePCate2u_slr20"] = DepthDamage.listOfdamage(Cate2depthu, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate2u_slr0"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr0"] 

Geodata["DamageCate2u_slr03"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr03"] 
Geodata["DamageCate2u_slr05"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr05"] 
Geodata["DamageCate2u_slr08"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr08"] 
Geodata["DamageCate2u_slr12"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr12"] 
Geodata["DamageCate2u_slr20"] = Geodata["Cost"]* Geodata["DamagePCate2u_slr20"] 

###############################################################################
Cate3depth = []
Cate3depth.append( np.round( Geodata["Cate3_slr0"] ).astype(int).tolist() )

Cate3depth.append( np.round( Geodata["Cate3_slr03"] ).astype(int).tolist() )
Cate3depth.append( np.round( Geodata["Cate3_slr05"] ).astype(int).tolist() )
Cate3depth.append( np.round( Geodata["Cate3_slr08"] ).astype(int).tolist() )
Cate3depth.append( np.round( Geodata["Cate3_slr12"] ).astype(int).tolist() )
Cate3depth.append( np.round( Geodata["Cate3_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate3_slr0"], Geodata["DamagePCate3_slr03"], Geodata["DamagePCate3_slr05"], Geodata["DamagePCate3_slr08"], Geodata["DamagePCate3_slr12"], Geodata["DamagePCate3_slr20"] = DepthDamage.listOfdamage(Cate3depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate3_slr0"] = Geodata["Cost"]* Geodata["DamagePCate3_slr0"] 

Geodata["DamageCate3_slr03"] = Geodata["Cost"]* Geodata["DamagePCate3_slr03"] 
Geodata["DamageCate3_slr05"] = Geodata["Cost"]* Geodata["DamagePCate3_slr05"] 
Geodata["DamageCate3_slr08"] = Geodata["Cost"]* Geodata["DamagePCate3_slr08"] 
Geodata["DamageCate3_slr12"] = Geodata["Cost"]* Geodata["DamagePCate3_slr12"] 
Geodata["DamageCate3_slr20"] = Geodata["Cost"]* Geodata["DamagePCate3_slr20"] 

###############################################################################
Cate3depthl = []
Cate3depthl.append( np.round( Geodata["Cate3l_slr0"] ).astype(int).tolist() )

Cate3depthl.append( np.round( Geodata["Cate3l_slr03"] ).astype(int).tolist() )
Cate3depthl.append( np.round( Geodata["Cate3l_slr05"] ).astype(int).tolist() )
Cate3depthl.append( np.round( Geodata["Cate3l_slr08"] ).astype(int).tolist() )
Cate3depthl.append( np.round( Geodata["Cate3l_slr12"] ).astype(int).tolist() )
Cate3depthl.append( np.round( Geodata["Cate3l_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate3l_slr0"], Geodata["DamagePCate3l_slr03"], Geodata["DamagePCate3l_slr05"], Geodata["DamagePCate3l_slr08"], Geodata["DamagePCate3l_slr12"], Geodata["DamagePCate3l_slr20"] = DepthDamage.listOfdamage(Cate3depthl, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)
Geodata["DamageCate3l_slr0"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr0"] 

Geodata["DamageCate3l_slr03"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr03"] 
Geodata["DamageCate3l_slr05"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr05"] 
Geodata["DamageCate3l_slr08"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr08"] 
Geodata["DamageCate3l_slr12"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr12"] 
Geodata["DamageCate3l_slr20"] = Geodata["Cost"]* Geodata["DamagePCate3l_slr20"] 

###############################################################################
Cate3depthu = []
Cate3depthu.append( np.round( Geodata["Cate3u_slr0"] ).astype(int).tolist() )

Cate3depthu.append( np.round( Geodata["Cate3u_slr03"] ).astype(int).tolist() )
Cate3depthu.append( np.round( Geodata["Cate3u_slr05"] ).astype(int).tolist() )
Cate3depthu.append( np.round( Geodata["Cate3u_slr08"] ).astype(int).tolist() )
Cate3depthu.append( np.round( Geodata["Cate3u_slr12"] ).astype(int).tolist() )
Cate3depthu.append( np.round( Geodata["Cate3u_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate3u_slr0"], Geodata["DamagePCate3u_slr03"], Geodata["DamagePCate3u_slr05"], Geodata["DamagePCate3u_slr08"], Geodata["DamagePCate3u_slr12"], Geodata["DamagePCate3u_slr20"] = DepthDamage.listOfdamage(Cate3depthu, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate3u_slr0"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr0"] 

Geodata["DamageCate3u_slr03"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr03"] 
Geodata["DamageCate3u_slr05"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr05"] 
Geodata["DamageCate3u_slr08"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr08"] 
Geodata["DamageCate3u_slr12"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr12"] 
Geodata["DamageCate3u_slr20"] = Geodata["Cost"]* Geodata["DamagePCate3u_slr20"] 

###############################################################################

Cate4depth = []
Cate4depth.append( np.round( Geodata["Cate4_slr0"] ).astype(int).tolist() )

Cate4depth.append( np.round( Geodata["Cate4_slr03"] ).astype(int).tolist() )
Cate4depth.append( np.round( Geodata["Cate4_slr05"] ).astype(int).tolist() )
Cate4depth.append( np.round( Geodata["Cate4_slr08"] ).astype(int).tolist() )
Cate4depth.append( np.round( Geodata["Cate4_slr12"] ).astype(int).tolist() )
Cate4depth.append( np.round( Geodata["Cate4_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate4_slr0"], Geodata["DamagePCate4_slr03"], Geodata["DamagePCate4_slr05"], Geodata["DamagePCate4_slr08"], Geodata["DamagePCate4_slr12"], Geodata["DamagePCate4_slr20"] = DepthDamage.listOfdamage(Cate4depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate4_slr0"] = Geodata["Cost"]* Geodata["DamagePCate4_slr0"] 

Geodata["DamageCate4_slr03"] = Geodata["Cost"]* Geodata["DamagePCate4_slr03"] 
Geodata["DamageCate4_slr05"] = Geodata["Cost"]* Geodata["DamagePCate4_slr05"] 
Geodata["DamageCate4_slr08"] = Geodata["Cost"]* Geodata["DamagePCate4_slr08"] 
Geodata["DamageCate4_slr12"] = Geodata["Cost"]* Geodata["DamagePCate4_slr12"] 
Geodata["DamageCate4_slr20"] = Geodata["Cost"]* Geodata["DamagePCate4_slr20"] 

###############################################################################
Cate4depthl = []
Cate4depthl.append( np.round( Geodata["Cate4l_slr0"] ).astype(int).tolist() )

Cate4depthl.append( np.round( Geodata["Cate4l_slr03"] ).astype(int).tolist() )
Cate4depthl.append( np.round( Geodata["Cate4l_slr05"] ).astype(int).tolist() )
Cate4depthl.append( np.round( Geodata["Cate4l_slr08"] ).astype(int).tolist() )
Cate4depthl.append( np.round( Geodata["Cate4l_slr12"] ).astype(int).tolist() )
Cate4depthl.append( np.round( Geodata["Cate4l_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate4l_slr0"], Geodata["DamagePCate4l_slr03"], Geodata["DamagePCate4l_slr05"], Geodata["DamagePCate4l_slr08"], Geodata["DamagePCate4l_slr12"], Geodata["DamagePCate4l_slr20"] = DepthDamage.listOfdamage(Cate4depthl, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate4l_slr0"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr0"] 

Geodata["DamageCate4l_slr03"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr03"] 
Geodata["DamageCate4l_slr05"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr05"] 
Geodata["DamageCate4l_slr08"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr08"] 
Geodata["DamageCate4l_slr12"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr12"] 
Geodata["DamageCate4l_slr20"] = Geodata["Cost"]* Geodata["DamagePCate4l_slr20"] 

###############################################################################
Cate4depthu = []

Cate4depthu.append( np.round( Geodata["Cate4u_slr0"] ).astype(int).tolist() )

Cate4depthu.append( np.round( Geodata["Cate4u_slr03"] ).astype(int).tolist() )
Cate4depthu.append( np.round( Geodata["Cate4u_slr05"] ).astype(int).tolist() )
Cate4depthu.append( np.round( Geodata["Cate4u_slr08"] ).astype(int).tolist() )
Cate4depthu.append( np.round( Geodata["Cate4u_slr12"] ).astype(int).tolist() )
Cate4depthu.append( np.round( Geodata["Cate4u_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate4u_slr0"], Geodata["DamagePCate4u_slr03"], Geodata["DamagePCate4u_slr05"], Geodata["DamagePCate4u_slr08"], Geodata["DamagePCate4u_slr12"], Geodata["DamagePCate4u_slr20"] = DepthDamage.listOfdamage(Cate4depthu, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate4u_slr0"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr0"] 

Geodata["DamageCate4u_slr03"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr03"] 
Geodata["DamageCate4u_slr05"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr05"] 
Geodata["DamageCate4u_slr08"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr08"] 
Geodata["DamageCate4u_slr12"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr12"] 
Geodata["DamageCate4u_slr20"] = Geodata["Cost"]* Geodata["DamagePCate4u_slr20"] 

###############################################################################

Cate5depth = []
Cate5depth.append( np.round( Geodata["Cate5_slr0"] ).astype(int).tolist() )

Cate5depth.append( np.round( Geodata["Cate5_slr03"] ).astype(int).tolist() )
Cate5depth.append( np.round( Geodata["Cate5_slr05"] ).astype(int).tolist() )
Cate5depth.append( np.round( Geodata["Cate5_slr08"] ).astype(int).tolist() )
Cate5depth.append( np.round( Geodata["Cate5_slr12"] ).astype(int).tolist() )
Cate5depth.append( np.round( Geodata["Cate5_slr20"] ).astype(int).tolist() )

Geodata["DamagePCate5_slr0"], Geodata["DamagePCate5_slr03"], Geodata["DamagePCate5_slr05"], Geodata["DamagePCate5_slr08"], Geodata["DamagePCate5_slr12"], Geodata["DamagePCate5_slr20"] = DepthDamage.listOfdamage(Cate5depth, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate5_slr0"] = Geodata["Cost"]* Geodata["DamagePCate5_slr0"] 

Geodata["DamageCate5_slr03"] = Geodata["Cost"]* Geodata["DamagePCate5_slr03"] 
Geodata["DamageCate5_slr05"] = Geodata["Cost"]* Geodata["DamagePCate5_slr05"] 
Geodata["DamageCate5_slr08"] = Geodata["Cost"]* Geodata["DamagePCate5_slr08"] 
Geodata["DamageCate5_slr12"] = Geodata["Cost"]* Geodata["DamagePCate5_slr12"] 
Geodata["DamageCate5_slr20"] = Geodata["Cost"]* Geodata["DamagePCate5_slr20"] 

###############################################################################
Cate5depthl = []

Cate5depthl.append( np.round( Geodata["Cate5l_slr0"] ).astype(int).tolist() )

Cate5depthl.append( np.round( Geodata["Cate5l_slr03"] ).astype(int).tolist() )
Cate5depthl.append( np.round( Geodata["Cate5l_slr05"] ).astype(int).tolist() )
Cate5depthl.append( np.round( Geodata["Cate5l_slr08"] ).astype(int).tolist() )
Cate5depthl.append( np.round( Geodata["Cate5l_slr12"] ).astype(int).tolist() )
Cate5depthl.append( np.round( Geodata["Cate5l_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate5l_slr0"], Geodata["DamagePCate5l_slr03"], Geodata["DamagePCate5l_slr05"], Geodata["DamagePCate5l_slr08"], Geodata["DamagePCate5l_slr12"], Geodata["DamagePCate5l_slr20"] = DepthDamage.listOfdamage(Cate5depthl, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate5l_slr0"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr0"] 

Geodata["DamageCate5l_slr03"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr03"] 
Geodata["DamageCate5l_slr05"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr05"] 
Geodata["DamageCate5l_slr08"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr08"] 
Geodata["DamageCate5l_slr12"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr12"] 
Geodata["DamageCate5l_slr20"] = Geodata["Cost"]* Geodata["DamagePCate5l_slr20"] 

###############################################################################
Cate5depthu = []
Cate5depthu.append( np.round( Geodata["Cate5u_slr0"] ).astype(int).tolist() )

Cate5depthu.append( np.round( Geodata["Cate5u_slr03"] ).astype(int).tolist() )
Cate5depthu.append( np.round( Geodata["Cate5u_slr05"] ).astype(int).tolist() )
Cate5depthu.append( np.round( Geodata["Cate5u_slr08"] ).astype(int).tolist() )
Cate5depthu.append( np.round( Geodata["Cate5u_slr12"] ).astype(int).tolist() )
Cate5depthu.append( np.round( Geodata["Cate5u_slr20"] ).astype(int).tolist() )


Geodata["DamagePCate5u_slr0"], Geodata["DamagePCate5u_slr03"], Geodata["DamagePCate5u_slr05"], Geodata["DamagePCate5u_slr08"], Geodata["DamagePCate5u_slr12"], Geodata["DamagePCate5u_slr20"] = DepthDamage.listOfdamage(Cate5depthu, FirstFloorHt, 
                                 floodzoneids, A1inundation, 
                                 V1inundation, O1inundation)

Geodata["DamageCate5u_slr0"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr0"] 

Geodata["DamageCate5u_slr03"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr03"] 
Geodata["DamageCate5u_slr05"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr05"] 
Geodata["DamageCate5u_slr08"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr08"] 
Geodata["DamageCate5u_slr12"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr12"] 
Geodata["DamageCate5u_slr20"] = Geodata["Cost"]* Geodata["DamagePCate5u_slr20"] 

###############################################################################
Geodata["Risk_slr0"] =   ( (Geodata["DamageCate5_slr0"] + Geodata["DamageCate4_slr0"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr0"] + Geodata["DamageCate3_slr0"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr0"] + Geodata["DamageCate2_slr0"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr0"] + Geodata["DamageCate1_slr0"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr0"] =   ( (Geodata["DamageCate5u_slr0"] + Geodata["DamageCate4u_slr0"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr0"] + Geodata["DamageCate3u_slr0"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr0"] + Geodata["DamageCate2u_slr0"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr0"] + Geodata["DamageCate1u_slr0"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr0"] =   ( (Geodata["DamageCate5l_slr0"] + Geodata["DamageCate4l_slr0"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr0"] + Geodata["DamageCate3l_slr0"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr0"] + Geodata["DamageCate2l_slr0"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr0"] + Geodata["DamageCate1l_slr0"])*(0.1  - 0.04)   ) / 2.0


###############################################################################

Geodata["Risk_slr03"] =   ( (Geodata["DamageCate5_slr03"] + Geodata["DamageCate4_slr03"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr03"] + Geodata["DamageCate3_slr03"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr03"] + Geodata["DamageCate2_slr03"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr03"] + Geodata["DamageCate1_slr03"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr03"] =   ( (Geodata["DamageCate5u_slr03"] + Geodata["DamageCate4u_slr03"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr03"] + Geodata["DamageCate3u_slr03"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr03"] + Geodata["DamageCate2u_slr03"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr03"] + Geodata["DamageCate1u_slr03"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr03"] =   ( (Geodata["DamageCate5l_slr03"] + Geodata["DamageCate4l_slr03"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr03"] + Geodata["DamageCate3l_slr03"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr03"] + Geodata["DamageCate2l_slr03"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr03"] + Geodata["DamageCate1l_slr03"])*(0.1  - 0.04)   ) / 2.0


###############################################################################

Geodata["Risk_slr05"] =   ( (Geodata["DamageCate5_slr05"] + Geodata["DamageCate4_slr05"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr05"] + Geodata["DamageCate3_slr05"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr05"] + Geodata["DamageCate2_slr05"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr05"] + Geodata["DamageCate1_slr05"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr05"] =   ( (Geodata["DamageCate5u_slr05"] + Geodata["DamageCate4u_slr05"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr05"] + Geodata["DamageCate3u_slr05"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr05"] + Geodata["DamageCate2u_slr05"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr05"] + Geodata["DamageCate1u_slr05"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr05"] =   ( (Geodata["DamageCate5l_slr05"] + Geodata["DamageCate4l_slr05"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr05"] + Geodata["DamageCate3l_slr05"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr05"] + Geodata["DamageCate2l_slr05"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr05"] + Geodata["DamageCate1l_slr05"])*(0.1  - 0.04)   ) / 2.0

###############################################################################

Geodata["Risk_slr08"] =   ( (Geodata["DamageCate5_slr08"] + Geodata["DamageCate4_slr08"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr08"] + Geodata["DamageCate3_slr08"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr08"] + Geodata["DamageCate2_slr08"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr08"] + Geodata["DamageCate1_slr08"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr08"] =   ( (Geodata["DamageCate5u_slr08"] + Geodata["DamageCate4u_slr08"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr08"] + Geodata["DamageCate3u_slr08"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr08"] + Geodata["DamageCate2u_slr08"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr08"] + Geodata["DamageCate1u_slr08"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr08"] =   ( (Geodata["DamageCate5l_slr08"] + Geodata["DamageCate4l_slr08"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr08"] + Geodata["DamageCate3l_slr08"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr08"] + Geodata["DamageCate2l_slr08"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr08"] + Geodata["DamageCate1l_slr08"])*(0.1  - 0.04)   ) / 2.0

###############################################################################

Geodata["Risk_slr12"] =   ( (Geodata["DamageCate5_slr12"] + Geodata["DamageCate4_slr12"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr12"] + Geodata["DamageCate3_slr12"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr12"] + Geodata["DamageCate2_slr12"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr12"] + Geodata["DamageCate1_slr12"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr12"] =   ( (Geodata["DamageCate5u_slr12"] + Geodata["DamageCate4u_slr12"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr12"] + Geodata["DamageCate3u_slr12"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr12"] + Geodata["DamageCate2u_slr12"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr12"] + Geodata["DamageCate1u_slr12"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr12"] =   ( (Geodata["DamageCate5l_slr12"] + Geodata["DamageCate4l_slr12"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr12"] + Geodata["DamageCate3l_slr12"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr12"] + Geodata["DamageCate2l_slr12"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr12"] + Geodata["DamageCate1l_slr12"])*(0.1  - 0.04)   ) / 2.0

###############################################################################

Geodata["Risk_slr20"] =   ( (Geodata["DamageCate5_slr20"] + Geodata["DamageCate4_slr20"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4_slr20"] + Geodata["DamageCate3_slr20"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3_slr20"] + Geodata["DamageCate2_slr20"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2_slr20"] + Geodata["DamageCate1_slr20"])*(0.1  - 0.04)   ) / 2.0


Geodata["Risku_slr20"] =   ( (Geodata["DamageCate5u_slr20"] + Geodata["DamageCate4u_slr20"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4u_slr20"] + Geodata["DamageCate3u_slr20"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3u_slr20"] + Geodata["DamageCate2u_slr20"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2u_slr20"] + Geodata["DamageCate1u_slr20"])*(0.1  - 0.04)   ) / 2.0


Geodata["Riskl_slr20"] =   ( (Geodata["DamageCate5l_slr20"] + Geodata["DamageCate4l_slr20"])*(0.01 - 0.002) +
                      (Geodata["DamageCate4l_slr20"] + Geodata["DamageCate3l_slr20"])*(0.02 - 0.01)  +
                      (Geodata["DamageCate3l_slr20"] + Geodata["DamageCate2l_slr20"])*(0.04 - 0.02)  +
                      (Geodata["DamageCate2l_slr20"] + Geodata["DamageCate1l_slr20"])*(0.1  - 0.04)   ) / 2.0


###############################################################################
Geodata_buyout = Geodata.loc[  Geodata["LUC"] == 0 ]

print( Geodata_buyout["Risk_slr0"].sum() )
print( Geodata_buyout["Risku_slr0"].sum() )
print( Geodata_buyout["Riskl_slr0"].sum() )
print( Geodata_buyout.Cost.sum() )




Geodata_exist = Geodata.loc[  Geodata["LUC"] >= 0 ]


print( Geodata_exist["Risk_slr0"].sum() )


print( Geodata_exist["Risk_slr03"].sum() )
print( Geodata_exist["Risku_slr03"].sum() )
print( Geodata_exist["Riskl_slr03"].sum() )

print( Geodata_exist["Risk_slr05"].sum() )
print( Geodata_exist["Risku_slr05"].sum() )
print( Geodata_exist["Riskl_slr05"].sum() )

print( Geodata_exist["Risk_slr08"].sum() )
print( Geodata_exist["Risku_slr08"].sum() )
print( Geodata_exist["Riskl_slr08"].sum() )

print( Geodata_exist["Risk_slr12"].sum() )
print( Geodata_exist["Risku_slr12"].sum() )
print( Geodata_exist["Riskl_slr12"].sum() )


print( Geodata_exist["Risk_slr20"].sum() )
print( Geodata_exist["Risku_slr20"].sum() )
print( Geodata_exist["Riskl_slr20"].sum() )



################################################################################
new_df3p = Geodata
################################################################################

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
pmarks = []
divider = make_axes_locatable(ax)
new_df3p.plot(column='Risk', scheme="Quantiles", k = 5,
           legend=True,
           ax=ax, cmap='PuRd',     
           edgecolor="gray",  # Borderline color
           linewidth=0.01,
           legend_kwds={'loc':'upper right', 
                        'bbox_to_anchor':(1.0, 0.45), 
                        'fmt':'{:.0f}',
                        'markerscale':1.0, 
                        'title_fontsize':'medium', 
                        'fontsize':'medium'}
           )

leg1 = ax.get_legend()
# Set markers to square shape
for ea in leg1.legendHandles:
    ea.set_marker('s')
leg1.set_title("Flood risk of buildings\nin the mean model($)")

x, y, arrow_length = 0.85, 0.6, 0.12
ax.annotate('N', xy=(x,y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='k', width = 10, headwidth = 30),
                va='center',ha='center', fontsize = 40,
                xycoords= ax.transAxes)

ax.add_artist(
             ScaleBar( 100, dimension="si-length", 
             units="km", location="lower center", 
             length_fraction = 0.2) )

#ax.set_title('', fontsize= 28)
ax.set_axis_on()
# ctx.add_basemap(ax ,
#                 source='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
#                 zoom=8)
#ctx.add_basemap(ax ,source= ctx.providers.Stamen.TonerLite)
#plt.tight_layout()
plt.savefig('./floodrisk0.png',dpi=300, bbox_inches='tight', pad_inches=0)

################################################################################
################################################################################


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
pmarks = []
divider = make_axes_locatable(ax)
new_df3p.plot(column='Risku', scheme="userdefined", 
              classification_kwds={'bins': [ 804, 3922, 7814, 13312]},
           legend=True,
           ax=ax, cmap='PuRd',     
           edgecolor="gray",  # Borderline color
           linewidth=0.01,
           legend_kwds={'loc':'upper right', 
                        'bbox_to_anchor':(1.0, 0.45), 
                        'fmt':'{:.0f}',
                        'markerscale':1.0, 
                        'title_fontsize':'medium', 
                        'fontsize':'medium'}
           )

leg1 = ax.get_legend()
# Set markers to square shape
for ea in leg1.legendHandles:
    ea.set_marker('s')
leg1.set_title("Flood risk of buildings\nin the 95%CI model($)")

x, y, arrow_length = 0.85, 0.6, 0.12
ax.annotate('N', xy=(x,y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='k', width = 10, headwidth = 30),
                va='center',ha='center', fontsize = 40,
                xycoords= ax.transAxes)

ax.add_artist(
             ScaleBar( 100, dimension="si-length", 
             units="km", location="lower center", 
             length_fraction = 0.2) )

#ax.set_title('', fontsize= 28)
ax.set_axis_on()
# ctx.add_basemap(ax ,
#                 source='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
#                 zoom=8)
#ctx.add_basemap(ax ,source= ctx.providers.Stamen.TonerLite)
#plt.tight_layout()
plt.savefig('./floodrisk95.png',dpi=300, bbox_inches='tight', pad_inches=0)

################################################################################
################################################################################


fig, ax = plt.subplots(1, 1, figsize=(8, 5))
pmarks = []
divider = make_axes_locatable(ax)
new_df3p.plot(column='Riskl', scheme="userdefined", 
              classification_kwds={'bins': [ 804, 3922, 7814, 13312]},

           legend=True,
           ax=ax, cmap='PuRd',     
           edgecolor="gray",  # Borderline color
           linewidth=0.01,
           
           legend_kwds={'loc':'upper right', 
                        'bbox_to_anchor':(1.0, 0.45), 
                        'fmt':'{:.0f}',
                        'markerscale':1.26, 
                        'title_fontsize':'medium', 
                        'fontsize':'medium'}
           )

leg1 = ax.get_legend()
# Set markers to square shape
for ea in leg1.legendHandles:
    ea.set_marker('s')
leg1.set_title("Flood risk of buildings\nin the 5%CI model($)")

x, y, arrow_length = 0.85, 0.6, 0.12
ax.annotate('N', xy=(x,y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='k', width = 10, headwidth = 30),
                va='center',ha='center', fontsize = 40,
                xycoords= ax.transAxes)

ax.add_artist(
             ScaleBar( 100, dimension="si-length", 
             units="km", location="lower center", 
             length_fraction = 0.2) )

#ax.set_title('', fontsize= 28)
ax.set_axis_on()
# ctx.add_basemap(ax ,
#                 source='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
#                 zoom=8)
#ctx.add_basemap(ax ,source= ctx.providers.Stamen.TonerLite)
#plt.tight_layout()
plt.savefig('./floodrisk05.png',dpi=300, bbox_inches='tight', pad_inches=0)

################################################################################


