#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:24:29 2022

@author: yuhan
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import geopandas as gpd
import operator 
import matplotlib.pyplot as plt
from ReclassifyLandUse import ReC
from DamageEstimation import DepthDamage
import matplotlib.pyplot as plt
from InputDataProcess import preprocess

class LUSimulation:
    
    def __init__(self, Geodata, predYrs, deltaY, initialYrs, LandGrowth, path = r'./model_Galveston/output/ParcelGC_LUC_cali.shp.zip'):
        Geodata     = Geodata
        predYrs     = predYrs
        deltaY      = deltaY
        initialYrs  = initialYrs
        LandGrowth  = LandGrowth
        pth         = path

    
    def CalDevlopedNeighbors(self, row, GeoNeighbors, Geodata, lucode):
        if row in GeoNeighbors:
            subframei = Geodata[ Geodata['TARGET_FID'].isin( 
                             GeoNeighbors[ row ] ) ]
            return len( subframei.loc[ subframei["LUCode11re"] == lucode] ) * 1.0 / 16.0 
        else: 
            return 0

    def simulate( self ):
        GeoNeighbors = {}
        improvedValues  = [0]*len( Geodata)
        changes         = [0]*len( Geodata)
        predyears       = range(initialYrs, predYrs, deltaY)
        for ykth in range( len( predyears ) ):
            """" update probs from gbdt"""
            yeark             = predyears[ ykth ]
            landuseProb       = [0]*len( Geodata)
            NeighbiorLU       = [0]*len( Geodata)
            landuseType       = [0]*len( Geodata)
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
                    landuseProb[idx]       = row.predprobs']
                    maxlu                  = row.predLU']
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
            Geodata["rank"]        = Geodata.reset_index().index +1
            Geodata["PAreaCum"]    = Geodata["PArea"].cumsum()
            #Geodata[ ["LUCode11re", "PArea", "PAreaCum"] ]
            lengthi     = Geodata["PAreaCum"].searchsorted(LandGrowth, side='right') + 1
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
                  " LU Predict ", np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & 
                                                           (Geodata["LUC"] < 6), 
                                           "PArea" ]), " \ndifference ", 
                  np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                                      (Geodata["LUCode20re"] < 6), "PArea" ] ) -\
                  np.sum( Geodata.loc[ (Geodata["LUC"] > 0) & 
                                           (Geodata["LUC"] < 6), 
                                      "PArea" ]) 
                  )
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
                
            return Geodata2
                
    def save_to_shp( self, Geodata2 ):
        Geodata2.to_file(filename= pth, driver='ESRI Shapefile')

    def plot_shapefile( self ):
        Geodata.plot( )
        plt.show()









predYrs     = 2020
deltaY      = 5
initialYrs  = 2010
fp = "./gis_parcel_Galveston/Parcels_inputs_fixed_TF.shp"
Geodata0 = gpd.read_file(fp)
dataenv = preprocess(Geodata0)
Geodata = dataenv.process()
LandGrowth = (np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
                                  (Geodata["LUCode20re"] < 7), "PArea" ]) - 
                     np.sum( Geodata.loc[ (Geodata["LUCode11re"] > 0) & 
                                  (Geodata["LUCode11re"] < 7), "PArea" ]) ) / 1.5


Geodata["LUC"]      = Geodata['LUCode11re']
env = LUSimulation(Geodata , predYrs, deltaY, initialYrs, LandGrowth)

results = env.simulate()

env.plot_shapefile()

env.save_to_shp(results)

# if __name__ == "__main__":
    

#     def main():
#         predYrs     = 2020
#         deltaY      = 5
#         initialYrs  = 2010
#         fp = "./gis_parcel_Galveston/Parcels_inputs_fixed_TF.shp"
#         Geodata0 = gpd.read_file(fp)
#         dataenv = preprocess(Geodata0)
#         Geodata = dataenv.process()
#         LandGrowth = (np.sum( Geodata.loc[ (Geodata["LUCode20re"] > 0) & 
#                                                  (Geodata["LUCode20re"] < 7), "PArea" ]) - 
#                              np.sum( Geodata.loc[ (Geodata["LUCode11re"] > 0) & 
#                                                  (Geodata["LUCode11re"] < 7), "PArea" ]) ) / 1.5
        
#         env = LUSimulation(Geodata , predYrs, deltaY, initialYrs, LandGrowth)
        
#         results = env.simulate()
        
#         env.plot_shapefile()
        
#         env.save_to_shp(results)
        

#     main()


