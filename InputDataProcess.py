#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 23:54:19 2022

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
import matplotlib.pyplot as plt
import pickle
from pyogrio import read_dataframe


class preprocess:
    def __init__(self, Geodata):
        Geodata    = Geodata
        dictpth    = r'../inputs/saved_dictionary.pkl'
        pthSG      = r'../outputs/Safegraph_VisitCount.csv'

        pth11      = r"../inputs/Harzus_tool/flBldgStructDmgFn.csv"
        pth22      = r"../inputs/Harzus_tool/flBldgContDmgFn.csv"
        pth33      = r"../inputs/Harzus_tool/flBldgInvDmgFn.csv"
        pthpolicy  = r"../outputs/Policy_Zip.csv"

        

    def process( self ):
        with open( dictpth, 'rb' ) as f:
            GeoNeighbors = pickle.load(f)
        ###############################################################################
        ###############################################################################
        SafeGraph_galvagg = pd.read_csv(pthSG, encoding='utf-8')
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
        Geodata['VAL19IMP']  = Geodata['VAL19TOT'] - Geodata['VAL19LAND'] 
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
        ###############################################################################
        DamageFunc1 =pd.read_csv(pth11, encoding='utf-8')
        DamageFunc2 =pd.read_csv(pth22, encoding='utf-8')
        DamageFunc3 =pd.read_csv(pth33, encoding='utf-8')
        print ("Finish read data first part")
        ###############################################################################
        ############Issues with group by###############################################  
        Policy_Zip =pd.read_csv(pthpolicy, encoding='utf-8')
        Policy_Zip = Policy_Zip[ ["ZipCode", "policyCost", 
                                  "policyCount", "CBR", 
                                  "CBRStd", "elevationN"] ]
        Policy_Zip["ZipCode"] = pd.to_numeric(Policy_Zip["ZipCode"], 
                                              errors='coerce')
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
                                left_on='ZipCode', 
                                right_on='ZipCode', how='left')

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
        print ("Finish read data second part")
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
        print ("Finish read data third part")

        Geodata['NN1'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                            CalDevlopedNeighbors(x, 
                                                            GeoNeighbors, 1) )
        Geodata['NN2'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                            CalDevlopedNeighbors(x, 
                                                            GeoNeighbors, 2) )
        Geodata['NN3'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                            CalDevlopedNeighbors(x, 
                                                            GeoNeighbors, 3) )
        Geodata['NN4'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                            CalDevlopedNeighbors(x, 
                                                            GeoNeighbors, 4) )
        Geodata['NN5'] = Geodata['TARGET_FID'].apply( lambda x: 
                                                            CalDevlopedNeighbors(x, 
                                                            GeoNeighbors, 5) )


        print (Geodata['NN1'].value_counts() )
        print (Geodata['NN2'].value_counts() )
        print (Geodata['NN3'].value_counts() )
        print (Geodata['NN4'].value_counts() )
        print (Geodata['NN5'].value_counts() )
        Geodata_select = Geodata[ ['ACRES', 'PopTot', 'TotalUnit', 'Mobile', 'vacant',
                                   'MedHHInc', 'TotAge65', 'BelowHigh', 'BelPoverty',
                                   'Minority', 'WhitePer', 'DEM', 'Cate4', 'BeachDist',
                                   'HealthDist', 'ParkDist', 'SchoolDist', 'CoastDist',
                                   'WetlatDist', 'NN1', 'NN2', 'NN2', 'NN4', 'NN5'] ]

        """ GBDT regression """
        
        Geodata['NewGrowth'] = Geodata.apply( ChangeState , axis=1)


        X_train, X_test, y_train, y_test = train_test_split(Geodata_select, Geodata['NewGrowth'], 
                                                            test_size=0.4, random_state=0)
        #model training started
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth= 20), n_estimators= 1200)
        ada_clf.fit(X_train, y_train)
        #model training finished

        predictions             = ada_clf.predict(Geodata_select)
        predictions_prob        = ada_clf.predict_proba( Geodata_select )
        maxls_probs             = [ max( predicti ) for predicti in predictions_prob ]
        Geodata['maxprobs']     = maxls_probs
        Geodata['predLU']       = predictions
        def CalUrbanGrowth(row):
            if row['predLU'] == 0:
                return 1 - row['maxprobs']
            else:
                return row['maxprobs']
        Geodata['predprobs'] = Geodata.apply( CalUrbanGrowth , axis=1)
        print ("Finish read data all")
        return Geodata


    def CalDevlopedNeighbors( self, row, GeoNeighbors, lucode):
        if row in GeoNeighbors:
            subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                             GeoNeighbors[ row ] ) ]
            return len( subframei.loc[ subframei["LUCode11re"] == lucode] ) * 1.0 / 16.0 
        else: 
            return 0
        
    def ChangeState(self, row):
        if row['lu20re'] == 1 and row['lu20re'] != row['lu11re'] :
            return 1
        elif row['lu20re'] == 2 and row['lu20re'] != row['lu11re'] :
            return 2
        elif row['lu20re'] == 3 and row['lu20re'] != row['lu11re'] :
            return 3
        elif row['lu20re'] == 4 and row['lu20re'] != row['lu11re'] :
            return 4
        elif row['lu20re'] == 5 and row['lu20re'] != row['lu11re'] :
            return 5
        else: 
            return 0
        
        
if __name__ == "__main__": 
    def main():
        fp = "../inputs/data_export/Parcels_tf6_5_area_unique.shp"
        Geodata0 = read_dataframe(fp)
        dataenv = preprocess(Geodata0)
        Geodata = dataenv.process()  
        print("succeed!")
    main()
     
        
