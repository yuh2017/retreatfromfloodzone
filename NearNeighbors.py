#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:56:46 2022

@author: yuhan
"""
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points


class Neighbor:
    
    def nearest(row, geom_union, df1, df2, geom1_col='geometry', 
                geom2_col='geometry', src_column=None):
        """Find the nearest point and return the corresponding value from specified column."""
        # Find the geometry that is closest
        nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
        # Get the corresponding value from df2 (matching is based on the geometry)
        value = df2[nearest][src_column].get_values()[0]
        return value