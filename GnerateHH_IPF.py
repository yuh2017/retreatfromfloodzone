import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import geopandas as gpd


def ifpCal(M , rowsum, colsum):
    rowsumi = np.sum(M, axis=0) #4
    #print(rowsum)
    #print(rowsumi)
    factorRow = np.divide( rowsum , rowsumi )
    factorRow = np.nan_to_num(factorRow, nan= 0, posinf= -1)
    nrow = len(M) #16
    ncol = len(M[0]) #4
    for j in range(ncol):
        frj = factorRow[j]
        for i in range(nrow):
            M[ i, j ] = M[ i, j ] * frj
    colsumi = np.sum(M, axis=1) # 16
    factorCol = np.divide( colsum , colsumi )
    factorCol = np.nan_to_num(factorCol, nan=0, posinf=-1)
    for j in range(ncol):
        for i in range(nrow):
            fri = factorCol[i]
            M[ i, j ] = M[ i, j ] * fri
    return np.round( M )

########################################################################################################################

fp0 = "/Users/yuhan/Documents/PropertyBuyout/Galveston_model_old_results/inputs_csv/Selected_SocioResults.csv"
GalvCendata0 = pd.read_csv( fp0 )
census_sample1  = GalvCendata0[ [ "CensusBlock", 
                              "aisan", "white", "black", "Others", "PopTot",
                              "$35k-$40k", "$75k-$100k",  "$50k-$60k",  
                              "$150k-$200k", "$200kmore",  "$30k-$35k",   
                              "$40k-$50k", "$125k-$150k", "$45k-$50k", 
                              "$100k-$125k", "$20k-$24k", "$60k-$75k", 
                              "$25k-$30k", "$15k-$20k", "$10k-$15k", 
                              "Less$10k" ] ]
census_sample1['CensusBlock'] = census_sample1['CensusBlock'].apply(int).apply(str)

########################################################################################################################
fp = "/Users/yuhan/Documents/PropertyBuyout/Galveston_model_old_results/inputs_csv/GalvestonCensus.csv"
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


simulatedPop = []
totalhh = 0
for index, censusi in census_sample.iterrows() :
    ###################################################################################################################
    #totalPop = censusi.white + censusi.black + censusi.aisan + censusi.Others
    #break
    incomei = [ censusi["Less$10k"], censusi["$10k-$15k"], censusi["$15k-$20k"], 
               censusi["$20k-$24k"],
      censusi["$25k-$30k"], censusi["$30k-$35k"], censusi["$35k-$40k"],
      censusi["$40k-$50k"], censusi["$45k-$50k"], censusi["$50k-$60k"], 
      censusi["$60k-$75k"], censusi["$75k-$100k"], censusi["$100k-$125k"], 
      censusi["$125k-$150k"], censusi["$150k-$200k"], censusi["$200kmore"] ]
    totalhh = totalhh + np.sum( incomei )
    Racei0 = [ censusi.white,  censusi.black,
              censusi.aisan, censusi.Others ]
    totalPop = np.sum( Racei0 )
    print( totalPop )
    #break
    if totalPop > 0:
        if totalPop > 0:
            Racei = [ np.round( Rei / totalPop * sum(incomei) ) for Rei in Racei0]
        else:
            Racei = [ 0 for Rei in Racei0]
    
        arr = np.array( [ [  np.floor( iinci/len(Racei) )  ] * len(Racei)
                          for iinci in incomei] )
        for x in range(20):
            arr = ifpCal(arr, Racei, incomei)
        
        ###################################################################################################################
        totalPop2 = censusi.PopTot
        Educationi = [censusi.BelowHigh, censusi.PopTot - censusi.BelowHigh]
        totalPop2 = np.sum( Educationi )
    
        if totalPop2 > 0:
            Educationi = [ np.round( edi / totalPop2 * sum(incomei) ) for edi in Educationi]
        else:
            Educationi = [ 0 for edi in Educationi]
    
        arr2 = np.array( [ [  np.floor( ri/len(Educationi) )  ] * len(Educationi)
                           for ri in Racei] )
        for x in range(20):
            arr2 = ifpCal(arr2, Educationi, Racei)
            
        #break
        ###################################################################################################################
        for ri in range(len(Racei)):
            incri = arr [:, ri]
            eduri = arr2[ri]
            #print(ri)
            #print( np.sum(incri), " ", np.sum(incri) )
            
            if sum(incri) > 0:
                arr21 = np.array( [ [  np.ceil( ei/len(incri) )  ] * len(incri) for ei in eduri] )
            else:
                arr21 = np.array([[0] * len(incri) for ei in eduri])
            for x in range(30):
                arr22 = ifpCal(arr21, incri, eduri)
            #print( arr22 )
            print( "Number of pop ", np.sum(arr22) )
            nrow = len( arr22 )  # 16
            ncol = len( arr22[0] )  # 4
            for j in range(ncol):
                for i in range(nrow):
                    if arr22[i, j] > 0:
                        for indi in range( int(arr22[i, j]) ):
                            data= { "race": ri+1 , "education": i+1, 
                                   "Income": j+1, "censusBlk":  censusi.CensusBlk,
                                   
                                   "aisan": censusi.aisan, "white": censusi.white,
                                   "black": censusi.black, "Others": censusi.Others,
                                  
                                   
                                   
                                   }
                            simulatedPop.append( data )
Populationx = pd.DataFrame( simulatedPop )
Populationx.to_csv(r'/Users/yuhan/Documents/PropertyBuyout/Gradient_Boost_code/outputs/Galveston_SimHHs.csv', index=False)

########################################################################################################################

import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


Populationx = pd.read_csv(r'/Users/yuhan/Documents/PropertyBuyout/Gradient_Boost_code/outputs/Galveston_SimHHs.csv')

fp = "/Users/yuhan/Documents/PropertyBuyout/Gradient_Boost_code/inputs/gis_parcel_Galveston/Parcels_inputs_fixed_TF5.shp"
parcels = gpd.read_file(fp)

parcels['VAL19TOT']  =  pd.to_numeric(parcels.VAL19TOT.replace(',', ''), 
                                    errors='coerce')
parcels['VAL19LAND'] =  pd.to_numeric(parcels.VAL19LAND.replace(',', ''), 
                                    errors='coerce')
parcels['VAL19IMP']  =  pd.to_numeric(parcels.VAL19IMP.replace(',', ''), 
                                    errors='coerce')
parcels['VAL19TOT']  = parcels['VAL19TOT'].fillna(0)
parcels['VAL19LAND'] = parcels['VAL19LAND'].fillna(0)
parcels['VAL19IMP']  = parcels['VAL19IMP'].fillna(0)


parcels['VAL19TOT']  =  parcels['VAL19LAND'] + parcels['VAL19IMP'] 



fth2 = "/Users/yuhan/Documents/Risk_Analysis_2/Miami_survey/New_SurveyData.csv"
SurveyData = pd.read_csv( fth2 )
########################################################################################################################
def reclassifyHHincome( hhinc ):
    if hhinc == 1:
        return 1
    elif hhinc == 2:
        return 1
    elif hhinc == 3:
        return 1
    elif hhinc == 4:
        return 2
    elif hhinc == 5:
        return 2
    elif hhinc == 6:
        return 3
    elif hhinc == 7:
        return 3
    elif hhinc == 8:
        return 4
    elif hhinc == 9:
        return 4
    elif hhinc == 10:
        return 5
    elif hhinc == 11:
        return 5
    elif hhinc == 12:
        return random.sample( [6, 7] , 1)[0]
    elif hhinc == 13:
        return random.sample( [7, 8] , 1)[0]
    elif hhinc == 14:
        return 9
    elif hhinc == 15:
        return 10
    else:
        return random.sample( [11, 12] , 1)[0]


def classifyJV( JV ):
    if JV <= 50000:
        return 1
    elif JV <=  90000 and JV > 50000:
        return 2
    elif JV <=  200000 and JV > 90000:
        return 3
    elif JV <=  300000 and JV > 200000:
        return 4
    elif JV <=  400000 and JV > 300000:
        return 5
    elif JV <=  500000 and JV > 400000:
        return 6
    elif JV <=  750000 and JV > 500000:
        return 6
    elif JV <=  1000000 and JV > 750000:
        return 8
    elif JV > 1000000:
        return 9
    else:
        return 1

########################################################################################################################
Populationx["SurvIncome"] = Populationx.apply( lambda x: reclassifyHHincome(x['Income']), axis=1 )
parcels["SurvJV"]         = parcels.apply( lambda x: classifyJV(x['VAL19TOT']), axis=1 )

SurveyPropertyValue = SurveyData.groupby(['PropertyValue']).size().div(len(SurveyData))
Survey_type_probs   = SurveyData.groupby(['PropertyValue'])['Income'].value_counts().div( len(SurveyData)).div(SurveyPropertyValue, axis=0)
probs_typeidxs      = Survey_type_probs.index.tolist()
probs_typeprobs     = Survey_type_probs.values.tolist()
probDict            = Survey_type_probs.to_dict()




surveyDic = {}
for keyi in probDict:
    if keyi[0] in surveyDic:
        if keyi[1] not in surveyDic[ keyi[0] ]:
            surveyDic[ keyi[0] ] [keyi[1]]  = {}
            surveyDic[ keyi[0] ][ keyi[1] ] = probDict[keyi]
        else:
            print(keyi[0], " "+ keyi[1] + " "+ probDict[keyi])
    else:
        surveyDic[keyi[0]] = {}
        surveyDic[keyi[0]][keyi[1]] = {}
        surveyDic[keyi[0]][keyi[1]] = probDict[keyi]


surveyDic[10][1]  = 0.0
surveyDic[10][3]  = 0.0
surveyDic[10][2]  = 0.0
surveyDic[10][10] = 0.09523809523809523
surveyDic[10][8]  = 0.09523809523809523
surveyDic[10][12] = 0.42857142857142855
surveyDic[10][11] = 0.23809523809523808


def frame2Dic(selectedHHs2):
    selectedHHs3 = selectedHHs2.to_dict( 'records' )
    outputs = {}
    for hhi in selectedHHs3:
        if hhi['Income'] in outputs:
            outputs[ hhi['Income'] ].append( hhi )
        else:
            outputs[hhi['Income']] = []
            outputs[hhi['Income']].append(hhi)
    return outputs

censblkList  = parcels.CensusBlk.unique().tolist()
parcels_dict = parcels.to_dict(orient='index')


AllPopDic = frame2Dic(Populationx)
kkk = 0
start = time.time()
genData = []
for censusi in censblkList:
    selectedParcels =  parcels.loc[ parcels['CensusBlk'] == censusi ].reset_index().to_dict('records') # [parcelKey]['SurvJV']
    selectedHHs2    = Populationx.loc[Populationx["censusBlk"] == censusi].reset_index()
    selectedHHs3    = frame2Dic( selectedHHs2 )
    for parcelsi in selectedParcels:
        BuildValue      = parcelsi["SurvJV"]
        BuildList       = list(surveyDic[BuildValue].keys())
        BuildProbList   = list(surveyDic[BuildValue].values())
        IncomeSelected = np.random.choice( BuildList, p= BuildProbList )
        if (parcelsi['poiclass'] == None and parcelsi['LANDUSE'] != None):
            if len( [word for word in parcelsi['LANDUSE'].split(',') if word in ['CDO','RA','RL','TL'] ] ) >= 0:
                if IncomeSelected not in selectedHHs3 :
                    IncomeSelected = np.random.choice(BuildList, p= BuildProbList)
                if IncomeSelected in selectedHHs3:
                    randHHi = random.choice( selectedHHs3[ IncomeSelected ] )
                    parcelsi['Race']        = randHHi['race']
                    parcelsi['income']      = randHHi['Income']
                    parcelsi['EducationL']  = randHHi['education']
                    if np.int64(parcelsi['VAL19IMP'] )> 0:
                        parcelsi['unitPrice']   = parcelsi['VAL19TOT'] / parcelsi['area']
                        parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
                    else: 
                        parcelsi['unitPrice']   = 0.0
                        parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
                    parcelsi['CensusBloc'] = censusi
                    genData.append(parcelsi)
                    selectedHHs3[IncomeSelected].remove(randHHi)
                    if len(selectedHHs3[IncomeSelected]) == 0:
                        selectedHHs3.pop(IncomeSelected, None)
                        print( len( selectedHHs3 ) )
                else:
                    randHHi = random.choice( AllPopDic[ IncomeSelected ] )
                    parcelsi['Race']        = randHHi['race']
                    parcelsi['income']      = randHHi['Income']
                    parcelsi['EducationL']  = randHHi['education']
                    if np.int64(parcelsi['VAL19IMP'] ) > 0:
                        parcelsi['unitPrice']   = parcelsi['VAL19TOT'] / parcelsi['area']
                        parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
                    else: 
                        parcelsi['unitPrice']   = 0.0
                        parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
                    parcelsi['CensusBloc'] = censusi
                    genData.append(parcelsi)
                    #AllPopDic[IncomeSelected].remove(randHHi)
                    if len( AllPopDic[IncomeSelected]) == 0:
                        AllPopDic.pop(IncomeSelected, None)
            else:
                parcelsi['Race']        = 0
                parcelsi['income']      = 0
                parcelsi['EducationL']  = 0
                parcelsi['unitPrice']   = 0.0
                parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
                parcelsi['CensusBloc'] = censusi
                genData.append(parcelsi)
        else:
            parcelsi['Race']        = 0
            parcelsi['income']      = 0
            parcelsi['EducationL']  = 0
            parcelsi['unitPrice']   = 0.0
            parcelsi['LandPrice']   = np.int64( parcelsi['VAL19LAND'] ) / parcelsi['area']
            parcelsi['CensusBloc'] = censusi
            genData.append(parcelsi)
        kkk = kkk + 1
        if kkk % 10000 == 0:
            print( "The parcel processed " , kkk)
end = time.time()
print(end - start)
HouseholdsData = gpd.GeoDataFrame( genData )


fpw = "/Users/yuhan/Documents/PropertyBuyout/Gradient_Boost_code/inputs/gis_parcel_Galveston/Parcels_inputs_fixed_TF6.shp"
#HouseholdsData.to_csv(r'C:/ResearchFiles/Galveston/outputs/GeneratedHouseholdsMD.csv', index=False)
HouseholdsData.to_file(filename= fpw, driver='ESRI Shapefile')

########################################################################################################################

# HouseholdCensus = HouseholdsData.merge(census_sample, 
#                                      left_on='CensusBloc', right_on='CensusBlock').reset_index()

# HouseholdCensus.to_csv(r'C:/ResearchFiles/Galveston/outputs/GeneratedHouseholdsMD.csv', index=False)

#GeneratedHouseholdsMD = pd.read_csv(r'C:\Users\zanwa\OneDrive\Desktop\LUP_code_data\GeneratedHouseholdsMD.csv')


