#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:32:52 2023

@author: yuhan
"""
import numpy as np
class Insurer:
    def __init__( self ):
        paras = [0.0, 0.0 ]
    def parameter_zone( self, aboveBFE, zone ):
        paras = [0.0, 0.0 ] 
        a0_p  =  0
        b0_c  =  0
        if zone == "A" :
            if aboveBFE > 3 :
                a0_p = 0.24
            elif aboveBFE > 2 and aboveBFE <= 3 :
                a0_p = 0.3
            elif aboveBFE > 1 and aboveBFE <= 2 :
                a0_p = 0.42
            elif aboveBFE > 0 and aboveBFE <= 1 :
                a0_p = 0.71
            elif aboveBFE > 0 and aboveBFE <= -1 :
                a0_p = 1.78
            else:
                a0_p = 4.4
        elif zone == "V":
            if aboveBFE > 3:
                a0_p = 0.7
            elif (aboveBFE > 2 and aboveBFE <= 3):
                a0_p = 0.75
            elif (aboveBFE > 1 and aboveBFE <= 2): 
                a0_p = 1.0
            elif ( aboveBFE > 0 and aboveBFE <= 1 ): 
                a0_p = 1.27
            elif ( aboveBFE > 0 and aboveBFE <= -1): 
                a0_p = 1.75
            elif( aboveBFE > -1 and aboveBFE <= -2): 
                a0_p = 2.39
            else:
                a0_p = 3.41	
        else:
            if aboveBFE > 3 :
                a0_p = 0.1
            elif(aboveBFE > 2 and aboveBFE <= 3):
                a0_p = 0.14
            elif(aboveBFE > 1 and aboveBFE <= 2):
                a0_p = 0.24
            elif aboveBFE > 0 and aboveBFE <= 1 :
                a0_p = 0.35
            elif( aboveBFE > 0 and aboveBFE <= -1):
                a0_p = 0.71
            else:
                a0_p = 1.4
        paras[0] = a0_p * 1
        paras[1] = b0_c
        return paras
    
    def insuranceCost( self, Value, content, SFHA, eleHeight, BFE = 10.68 ):
        discounts = 1.0
        Azone = 0
        Vzone = 0
        if SFHA == 10:
            Vzone = 1
        elif SFHA in [3,5,6,8] :
            Azone = 1
        InsCosts = [0,0, 0,0, 0,0 ] 
        new_elevation = eleHeight 
        aboveBFE = int( np.round( new_elevation - BFE ) )
        
        if content > 100000.0:
            content = 100000.0
        
        Coverage = 0
        Insurance = 0 
        if( Value < 60000.0): 
            Coverage = 60000.0
        elif( Value >= 60000.0 and Value < 250000.0): 
            Coverage = Value
        else:
            Coverage = 250000.0
        if(Azone > 0 ): 
            zoneAV = "A" 
        elif( Vzone > 0 ): 
            zoneAV = "V" 
        else: 
            zoneAV = "NA" 
        #print( aboveBFE, zoneAV )
        paras = parameter_zone(  aboveBFE, zoneAV ) 
        a0_p = paras[0] 
        b0_c = paras[1] 
        Insurance = a0_p * Coverage/100.0 * discounts + b0_c*content * discounts
        InsCosts = [ Coverage, Insurance ] 
        return InsCosts