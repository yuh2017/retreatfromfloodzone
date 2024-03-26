import operator 
import numpy as np
'''
0.0 
5.42e-5
1.74e-4
3.12e-4
'''

#2.68
#2.97
#3.18
#3.39
#3.61
import numpy as pd
class DepthDamage:
    def __init__( self, DamageFunc1, DamageFunc2, DamageFunc3 ):
        #pth1 = r"../inputs/Harzus_tool/flBldgStructDmgFn.csv"
        #pth2 = r"../inputs/Harzus_tool/flBldgContDmgFn.csv"
        #pth3 = r"../inputs/Harzus_tool/flBldgInvDmgFn.csv"
        self.DamageFunc1     = DamageFunc1
        self.DamageFunc2     = DamageFunc2
        self.DamageFunc3     = DamageFunc3
        print ("Finish read data second part")
        A1              = self.DamageFunc1.loc[ self.DamageFunc1.Description == 'one floor, no basement, Structure, A-Zone' ].reset_index().loc[0].tolist()[5:34]
        V1              = self.DamageFunc1.loc[ self.DamageFunc1.Description == 'one floor, no basement, Structure, V-Zone' ].reset_index().loc[0].tolist()[5:34]
        O1              = self.DamageFunc1.loc[ self.DamageFunc1.Description == 'one story, no basement, Structure' ].reset_index().loc[0].tolist()[5:34]
        inundation      = list( range(-4, 25 ) )
        self.A1inundation    = dict(zip(inundation, A1))
        self.V1inundation    = dict(zip(inundation, V1))
        self.O1inundation    = dict(zip(inundation, O1))
    
    def catedepth(self, catei, stormheight):
        if catei < 0:
            catei = 0.0
        #if catei > stormheight * 3.32 :
        #    catei = stormheight* 3.32
        return catei

    def f(self, dem):
        b0_sd = 0.09179
        b1_sd = 0.00963
        return 5.5856 - 0.4401 * dem * 0.302
    
    def damage(self, inundation, FFH, floodPlain):
        floodheight = list( map(operator.sub, inundation, FFH) )
        damagefunc1 = self.A1inundation
        damagefunc2 = self.V1inundation
        damagefunc3 = self.O1inundation
        damagelist = []
        for idx in range( len(floodheight) ):        
            fhi = floodheight[idx]
            fldplainid = floodPlain[idx]
            if fldplainid <= 5:
                damagefunc = damagefunc1
            elif fldplainid == 9 or fldplainid == 10:
                damagefunc = damagefunc2
            else:
                damagefunc = damagefunc3
            if fhi >= -4 and fhi < 25:
                damagelist.append(  damagefunc[ fhi ]* 1.0/ 100.0 )
            elif fhi < -4:
                damagelist.append(  0.0 )
            elif fhi >= 25:
                damagelist.append( 1.0 )
        return damagelist
    
    def damage_heighti(self, floodinundation, FFH, floodPlain ):
        fhi = np.int8( np.round( floodinundation - FFH ) )
        fldplainid = floodPlain
        damagefunc1 = self.A1inundation
        damagefunc2 = self.V1inundation
        damagefunc3 = self.O1inundation
        if fldplainid <= 5:
            damagefunc = damagefunc1
        elif fldplainid == 9 or fldplainid == 10:
            damagefunc = damagefunc2
        else:
            damagefunc = damagefunc3
        if fhi <= 0:
            damagei = 0
        elif fhi > 0 and fhi < 25:
            damagei = damagefunc[ fhi ]* 1.0/ 100.0 
        elif fhi >= 25:
            damagei = 1.0
        return damagei

    def listOfdamage(self, inundations, FFH, floodPlain ):
        listOfdamagelist = []
        damagefunc1 = self.A1inundation
        damagefunc2 = self.V1inundation
        damagefunc3 = self.O1inundation
        for inundation in inundations:
            floodheight = list( map(operator.sub, inundation, FFH) )
            damagelist = []
            for idx in range( len(floodheight) ):        
                fhi = floodheight[idx]
                fldplainid = floodPlain[idx]
                if fldplainid <= 5:
                    damagefunc = damagefunc1
                elif fldplainid == 9 or fldplainid == 10:
                    damagefunc = damagefunc2
                else:
                    damagefunc = damagefunc3
                if fhi >= -4 and fhi < 25:
                    damagelist.append(  damagefunc[ fhi ]* 1.0/ 100.0 )
                elif fhi < -4:
                    damagelist.append(  0.0 )
                elif fhi >= 25:
                    damagelist.append( 1.0 )
            listOfdamagelist.append( damagelist )
        return listOfdamagelist
    
    def floodh_adjust(self, probi , floodx, lgwaterdist ):
        if probi >= 0.1:
            return 0.505721 + 0.271128 * floodx  - 0.07561 * lgwaterdist
        elif probi < 0.1 and probi >= 0.04:
            return 1.831984 + 0.822664 * floodx  - 0.170123  * lgwaterdist
        elif probi < 0.04 and probi >= 0.02:
            return 5.114833 + 0.967940 * floodx  - 0.399485 * lgwaterdist
        elif probi < 0.02 and probi >= 0.01:
            return 7.808424 + 0.977928 * floodx  - 0.495649 * lgwaterdist
        else:
            return 10.446903 + 0.900420 * floodx - 0.496787 * lgwaterdist

    def dem_waterdist(self, delta_dem ):
        waterdist = 49.7718 * delta_dem
        return waterdist
            


    def RiskCalculation (self, FirstFloor, floodzones,
                         y_from, y_end, waterdist, dem, cate_diffs, 
                         slrH = 0, para_b = 0.000003, mui = 1995.194 , sigmai = 304.512):
        cateprobs    = np.array( [0.9, 0.96, 0.98, 0.99, 0.995]) 
        #cate_heights = np.zeros( len(cateprobs) )
        cate_damages = np.zeros( len(cateprobs) ) 
        #slrH         = 0.0
        totalRisk    = 0
        for yith in range( y_from, y_end ):
            if yith - y_from >= 1 and para_b > 0:
                rate = ( 0.0017*( yith ) + para_b*( ( yith )** 2) ) - ( 0.0017*( yith -1) + para_b*( ( yith -1) ** 2) )
            else:
                rate = 0 
            slrH = ( slrH + rate ) 
            slrH_mm = slrH* 1000
            #if yith % 5 == 0:
            #    print( yith, " rate ", rate, " slr", slrH_mm )
            distchange = self.dem_waterdist( (slrH_mm ) / 1000 ) 
            #print("distchange",distchange)
            logdisti = np.log( waterdist - distchange ) if waterdist > distchange else 0
            cate_id      = 0
            for cateip in cateprobs:
                cate_heighti = ( mui + slrH_mm - 1 * np.log( -1 * np.log( cateip ) )   * sigmai ) / 304.8
                if dem < cate_heighti and dem > 0:
                    cate_floodi  = cate_heighti - dem  
                elif dem < 0 :
                    cate_floodi  = cate_heighti
                else:
                    cate_floodi  = 0
                if cate_diffs[-1] > 0:
                    cate_heightsi = self.floodh_adjust( 1 - cateip , cate_floodi, logdisti )
                else:
                    cate_heightsi = cate_floodi
                cate_damages[cate_id] = self.damage_heighti( cate_heightsi + cate_diffs[cate_id], 
                                                             FirstFloor , floodzones )
                cate_id = cate_id + 1
            risk_ith = ( 
                     ( cate_damages[4] + cate_damages[3] ) * (0.01 - 0.002)  +
                     ( cate_damages[3] + cate_damages[2] ) * (0.02 - 0.01)   +
                     ( cate_damages[2] + cate_damages[1] ) * (0.04 - 0.02)   +
                     ( cate_damages[1] + cate_damages[0] ) * (0.1  - 0.04)   ) / 2.0
            #if yith % 5 == 0:
            #    print( yith, " total risk ", risk_ith, " damage ", cate_damages  )
            totalRisk = totalRisk + risk_ith
        return totalRisk