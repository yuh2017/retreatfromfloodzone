'''
land use code
0 water   0
1 developed open space   1
2 developed low   1
3 developed medium   1
4 developed high   1
5 barren land   2
6 forest/psature/scurb   3
7 cultivated crops   4
8 wetland   5

Building Code reclassified
"Residential"   : 1
"Commercial"    : 2
"Industrial"    : 3
"Golf"          : 6
"Agriculture"   : 6
"GreenBelt"     : 6
"Infrastructure": 4
"Public"        : 5
"Undeveloped"   : 7
"UnderWater"    : 6
"GreenLand"     : 6
"WetLand"       : 6
"other"         : 6
'''

class ReC:
    def reclassify(row):   # row is a Series
        if row == 0:
            return 0
        elif row == 1 : 
            return 1
        elif row == 2 or row == 3 or row == 4:
            return 1
        elif row == 5:
            return 2
        elif row == 6:
            return 3
        elif row == 7:
            return 4
        elif row == 8:
            return 5
        else:
            return -1


    def reassignNonLU(row):   # row is a Series
    #and any(substring in row['LANDUSE'] for substring in ["CDO","R1","RA","RL","TL"])
        if row.lu20re == 1 and row.VAL19IMP <= 0 :
            row.lu06re = 7
            row.lu11re = 7
            row.lu15re = 7
            row.lu20re = 7
        if row[ 'landCode' ] == "Undeveloped" and row.VAL19IMP <= 0:
            row.lu06re = 7
            row.lu11re = 7
            row.lu15re = 7
            row.lu20re = 7
        if row[ 'landCode' ] == "Agriculture" and ( row.lu06re == 0 or row.lu06re == 5):
            row.lu06re = 9
        if row[ 'landCode' ] == "Agriculture" and ( row.lu11re == 0 or row.lu11re == 5):
            row.lu11re = 9
        if row[ 'landCode' ] == "Agriculture" and ( row.lu15re == 0 or row.lu15re == 5):
            row.lu15re = 9
        if row[ 'landCode' ] == "Agriculture" and ( row.lu20re == 0 or row.lu20re == 5): 
            row.lu20re = 9
        #print (row['LANDUSE'])
        return row
    '''
    residential     = "CDO|DKM|R1|RA|RH|RL|TL|PF|SF"
    nonresidential  = "FR|PU|RF|RL"
    commercial      = "CL|CO"                 
    
    Industrial      = "IND" 
    Golf            = "GC|GF"
    Agriculture     = "A1|A2|A3|AG|AGE|B1|B2|C1|D1|DC|E1|E2|E3|E4|E5|E6|F1|F2|F3|F5|F6|F7|F9|IC|IP|NP|R1"
    
    GreenLand       = "F4"
    WetLand         = "ML"
    GreenBelt       = "GB"
    
    Pline           = "PL|PWL|DE" 
    Road            = "FR|RF|RW" 
    Public          = "PU"
    
    restricted      = "RU"
    greenBelt       = "GB"
    Undeveloped     = "UN"
    underwater      = "UW"
    '''
    def reassignBuildingCode(row):   # row is a Series
    #and any(substring in row['LANDUSE'] for substring in ["CDO","R1","RA","RL","TL"])
        residential     = "CDO|DKM|R1|RA|RH|RL|TL|PF|SF"
        nonresidential  = "FR|PU|RF|RL"
        commercial      = "CL|CO"                 
        Industrial      = "IND" 
        Golf            = "GC|GF"
        Agriculture     = "A1|A2|A3|AG|AGE|B1|B2|C1|D1|DC|E1|E2|E3|E4|E5|E6|F1|F2|F3|F5|F6|F7|F9|IC|IP|NP|R1"
        GreenLand       = "F4"
        WetLand         = "ML"
        GreenBelt       = "GB"
        Pline           = "PL|PWL|DE" 
        Road            = "FR|RF|RW" 
        Public          = "PU"
        restricted      = "RU"
        Undeveloped     = "UN"
        underwater      = "UW"

        if row in residential:
            return "Residential"
        elif row in commercial:
            return "Commercial"
        elif row in Industrial:
            return "Industrial"
        elif row in Golf:
            return "Golf"
        elif row in Agriculture:
            return "Agriculture"
        elif row in GreenBelt:
            return "GreenBelt"
        elif row in Road or row in Pline:
            return "Infrastructure"
        elif row in Public:
            return "Public"
        elif row in Undeveloped:
            return "Undeveloped"
        elif row in underwater:
            return "UnderWater"
        elif row in GreenLand:
            return "GreenLand"
        elif row in WetLand:
            return "WetLand"
        else:
            return "Other"

    def reassignBuildingCode2(row):
        if row == "Residential":
            return 1
        elif row  == "Commercial":
            return 2
        elif row  == "Industrial":
            return 3
        elif row  == "Golf":
            return 6
        elif row  == "Agriculture":
            return 6
        elif row  == "GreenBelt":
            return 6
        elif row  == "Infrastructure":
            return 4
        elif row  == "Public":
            return 5
        elif row  == "Undeveloped":
            return 7
        elif row  == "UnderWater":
            return 6
        elif row  == "GreenLand":
            return 6
        elif row  == "WetLand":
            return 6
        elif row  == "Other":
            return 6

    def reclassify2(row):   # row is a Series
        if row.lu01re != 1 and row.LUCode01re > 0 and row.LUCode01re <= 5: 
            row.LUCode01re = 7
        elif row.lu01re != 1 and row.LUCode01re > 5:
            row.LUCode01re = 6
        
        if row.lu06re != 1 and row.LUCode06re > 0 and row.LUCode06re <= 5: 
            #row.LUCode01re = 7
            row.LUCode06re = 7
        elif row.lu06re != 1 and row.LUCode06re > 5:
            row.LUCode06re = 6
        
        if row.lu11re != 1 and row.LUCode11re > 0 and row.LUCode11re <= 5: 
            #row.LUCode01re = 7
            #row.LUCode06re = 7
            row.LUCode11re = 7
        elif row.lu11re != 1 and row.LUCode11re > 5:
            row.LUCode11re = 6
        
        if row.lu15re != 1 and row.LUCode15re > 0 and row.LUCode15re <= 5: 
            #row.LUCode01re = 7
            #row.LUCode06re = 7
            #row.LUCode11re = 7
            row.LUCode15re = 7
        elif row.lu15re != 1 and row.LUCode15re > 5:
            row.LUCode15re = 6
        
        if row.lu20re != 1 and row.LUCode20re > 0 and row.LUCode20re <= 5: 
            #row.LUCode01re = 7
            #row.LUCode06re = 7
            #row.LUCode11re = 7
            #row.LUCode15re = 7
            row.LUCode20re = 7
        elif row.lu20re != 1 and row.LUCode20re > 5:
            row.LUCode20re = 6
        #print (row)
        return row
    ###############################################################################