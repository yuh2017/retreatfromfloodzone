library(ggplot2)
library(ggregplot)
library(tidyverse)
library(RColorBrewer)
library(SpatialEpi)
library(leaflet)
library(sp)
library(spdep)
library(hrbrthemes)
library(ggpmisc)
############################################################################
#poi_pth <- "/Users/yuhan/Desktop/Gradient_Boost_code/csv/parcel_poi_tf6_6.csv"
#poiData   <- read.csv(file= poi_pth, header=TRUE)


parcel_pthi  <- "/Users/yuhan/Desktop/Gradient_Boost_code/csv/sce1.csv" 
ParcelData   <- read.csv(file= parcel_pthi, header=TRUE)

ParcelData1 <- ParcelData[ ParcelData$Cate1< 22, ]


############################################################################

cate5avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 )
WaterCoastDist5 <- c( mean( ParcelData1[ ParcelData1 $Cate5 == 1, "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 2,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 3,   "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 4,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 5,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 6,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 7,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 8,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 9,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 10, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 11, "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 12, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 13, "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 14, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 15, "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 16, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 17, "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 18, "WaterCoastDist" ] ) )


############################################################################
cate4avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
WaterCoastDist4 <- c( 
mean( ParcelData1[ ParcelData1 $Cate4 == 1, "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 2,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate4 == 3,   "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 4,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate4 == 5,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 6,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate4 == 7,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 8,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate4 == 9,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 10, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate4 == 11, "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 12, "WaterCoastDist" ] ) 
)


############################################################################
cate3avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11)
WaterCoastDist3 <- c( mean( ParcelData1[ ParcelData1 $Cate3 == 1, "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate3 == 2,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 3,   "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate3 == 4,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 5,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 6,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 7,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 8,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 9,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 10, "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 11, "WaterCoastDist" ] ) )


############################################################################
cate2avg <- c( 1, 2,3, 4, 5, 6, 7, 8 )
WaterCoastDist2 <- c( mean( ParcelData1[ ParcelData1 $Cate2 == 1, "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate2 == 2,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 3,   "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate2 == 4,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 5,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate2 == 6,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 7,   "WaterCoastDist" ] ),
mean( ParcelData1[ ParcelData1 $Cate2 == 8,   "WaterCoastDist" ] ) )


############################################################################
cate1avg <- c( 1, 2,3 )
WaterCoastDist1 <- c( mean( ParcelData1[ ParcelData1 $Cate1 == 1, "WaterCoastDist" ] ), 
mean( ParcelData1[ ParcelData1 $Cate1 == 2,   "WaterCoastDist" ] ), mean( ParcelData1[ ParcelData1 $Cate1 == 3,   "WaterCoastDist" ] ) )


############################################################################

cate5avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 )
DEM5avg <- c( 
mean( ParcelData1[ ParcelData1 $Cate5 == 1, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 2,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 3,   "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 4,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 5,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 6,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 7,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 8,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 9,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 10, "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 11, "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 12, "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 13, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 14, "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 15, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate5 == 16, "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate5 == 17, "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate5 == 18, "DEM" ] ) )



############################################################################

cate4avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )
DEM4avg <- c( 
mean( ParcelData1[ ParcelData1 $Cate4 == 1, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 2, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 3, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 4, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 5, "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 6, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 7, "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 8, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 9, "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 10,"DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate4 == 11,"DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate4 == 12,"DEM" ] ) )



############################################################################

cate3avg <- c( 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11 )
DEM3avg <- c( 
mean( ParcelData1[ ParcelData1 $Cate3 == 1, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate3 == 2,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 3,   "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate3 == 4,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 5,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 6,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 7,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 8,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 9,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate3 == 10, "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate3 == 11, "DEM" ] ) )

############################################################################

cate2avg <- c( 1, 2,3, 4, 5, 6, 7, 8 )
DEM2avg <- c( 
mean( ParcelData1[ ParcelData1 $Cate2 == 1, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate2 == 2,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 3,   "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate2 == 4,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 5,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate2 == 6,   "DEM" ] ), mean( ParcelData1[ ParcelData1 $Cate2 == 7,   "DEM" ] ),
mean( ParcelData1[ ParcelData1 $Cate2 == 8,   "DEM" ] ) )



############################################################################

cate1avg <- c( 1, 2,3 )
DEM1avg <- c( 
mean( ParcelData1[ ParcelData1 $Cate1 == 1, "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate1 == 2,   "DEM" ] ), 
mean( ParcelData1[ ParcelData1 $Cate1 == 3,   "DEM" ] ) )

############################################################################
floodx1 <- ifelse ( DEM1avg < 2650.643423 / 0.3048, 2650.643423 /1000 /0.3048 - DEM1avg, 0 )
floodx2 <- ifelse ( DEM2avg < 2650.643423 / 0.3048, 2899.025313 /1000 /0.3048 - DEM2avg, 0 )
floodx3 <- ifelse ( DEM3avg < 2650.643423 / 0.3048, 3143.795993 /1000 /0.3048 - DEM3avg, 0 )
floodx4 <- ifelse ( DEM4avg < 2650.643423 / 0.3048, 3451.614502 /1000 /0.3048 - DEM4avg, 0 )
floodx5 <- ifelse ( DEM5avg < 2650.643423 / 0.3048, 3838.720651 /1000 /0.3048 - DEM5avg, 0 )

summary( lm(cate1avg ~ floodx1 + log( WaterCoastDist1 ) ) )
summary( lm(cate2avg ~ floodx2 + log( WaterCoastDist2 ) ) )
summary( lm(cate3avg ~ floodx3 + log( WaterCoastDist3 ) ) )
summary( lm(cate4avg ~ floodx4 + log( WaterCoastDist4 ) ) )
summary( lm(cate5avg ~ floodx5 + log( WaterCoastDist5 ) ) )

my_data1 <- data.frame(cate1avg, floodx1, WaterCoastDist1)  
colnames( my_data1 ) <- c( "cate", "floodx", "waterdist")
my_data2 <- data.frame(cate2avg, floodx2, WaterCoastDist2) 
colnames( my_data2 ) <- c( "cate", "floodx", "waterdist")
   
my_data3 <- data.frame(cate3avg, floodx3, WaterCoastDist3)    
colnames( my_data3 ) <- c( "cate", "floodx", "waterdist")

my_data4 <- data.frame(cate4avg, floodx4, WaterCoastDist4)    
colnames( my_data4 ) <- c( "cate", "floodx", "waterdist")

my_data5 <- data.frame(cate5avg, floodx5, WaterCoastDist5)    
colnames( my_data5 ) <- c( "cate", "floodx", "waterdist")

total_data <- bind_rows( my_data3, my_data4, my_data5 )
#Wrong
summary( lm(cate ~ floodx + log( waterdist ) , data = total_data) )
#Corrected
summary( lm(cate ~ floodx + log( waterdist ) , data = my_data4) )



cate4avg_all <- c( cate1avg, cate2avg, cate3avg, cate4avg, cate5avg)
Floodavg_all  <- c(  floodx1, floodx2, floodx3, floodx4, floodx5)
WaterCoastDist_all <- c( WaterCoastDist1, WaterCoastDist2, WaterCoastDist3, WaterCoastDist4, WaterCoastDist5)

my_data0 <- data.frame(cate4avg_all, Floodavg_all, WaterCoastDist_all)    
# Apply data.frame function
my_data0$cate4avg_all <- my_data0$cate4avg_all * 0.3048
my_data0$Floodavg_all <- my_data0$Floodavg_all * 0.3048
#my_data0$WaterCoastDist_all <-  log( my_data0$WaterCoastDist_all ) 

summary( lm( cate4avg_all ~ Floodavg_all +  WaterCoastDist_all   , data = my_data0 ) )

############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################


x <- c( 2650.643423 /1000 /0.3048 , 2899.025313 /1000 /0.3048 , 3143.795993 /1000 /0.3048 , 
	3451.614502 /1000 /0.3048 , 3838.720651/1000 /0.3048 )
y <- c( mean( ParcelData$Cate1 ) * 0.3048, mean( ParcelData$Cate2 ) * 0.3048, 
	mean( ParcelData$Cate3 ) * 0.3048, mean( ParcelData$Cate4 ) * 0.3048, 
	mean( ParcelData$Cate5 ) * 0.3048   )
plot( x, y )


x1 <- ifelse ( ParcelData1 $DEM < 2650.643423 / 0.3048, 2650.643423/ 0.3048 - ParcelData1 $DEM, 0 )
x2 <- ParcelData1 $WaterCoastDist
x3 <- ParcelData1 $DEM
y1 <- ParcelData1 $Cate1 

my_data <- data.frame(y1, x1, x2, x3)    # Apply data.frame function
my_data$x1 <- x1* 0.3048

############################################################################################################
plot(x1, y1)
yavg1 <- c(0, 1, 2,3, 4, 5, 6, 7)
xavg1 <- c( mean( my_data[ my_data$y1 == 0, "x1" ] ), mean( my_data[ my_data$y1 == 1, "x1" ] ), 
mean( my_data[ my_data$y1 == 2, "x1" ] ), mean( my_data[ my_data$y1 == 3, "x1" ] ), 
mean( my_data[ my_data$y1 == 4, "x1" ] ), mean( my_data[ my_data$y1 == 5, "x1" ] ),
mean( my_data[ my_data$y1 == 6, "x1" ] ), mean( my_data[ my_data$y1 == 7, "x1" ] ) )
plot( xavg1, yavg1 )

############################################################################################################

x2 <- ifelse ( ParcelData1 $DEM < 2899.025313 / 0.3048, 2899.025313/ 0.3048 - ParcelData1 $DEM, 0 )
y2 <- ParcelData1 $Cate2
plot(x2, y2)

my_data$x2 <- x2* 0.3048/1000
my_data$y2 <- y2

yavg2 <- c(0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
xavg2 <- c( mean( my_data[ my_data$y2 == 0, "x2" ] ), 
mean( my_data[ my_data$y2 == 1, "x2" ] ), 
mean( my_data[ my_data$y2 == 2, "x2" ] ), 
mean( my_data[ my_data$y2 == 3, "x2" ] ), 
mean( my_data[ my_data$y2 == 4, "x2" ] ), 
mean( my_data[ my_data$y2 == 5, "x2" ] ),
mean( my_data[ my_data$y2 == 6, "x2" ] ), 
mean( my_data[ my_data$y2 == 7, "x2" ] ),
mean( my_data[ my_data$y2 == 8, "x2" ] ), 
mean( my_data[ my_data$y2 == 9, "x2" ] ),
mean( my_data[ my_data$y2 == 10, "x2" ] ), 
mean( my_data[ my_data$y2 == 11, "x2" ] ),
mean( my_data[ my_data$y2 == 12, "x2" ] ) )
plot( xavg2, yavg2 )


############################################################################################################

x3 <- ifelse ( ParcelData1 $DEM < 3143.795993 / 0.3048,  3143.795993/ 0.3048 - ParcelData1 $DEM, 0 )
y3 <- ParcelData1 $Cate3
plot(x3, y3)

my_data$x3 <- x3* 0.3048/1000
my_data$y3 <- y3

yavg3 <- c(0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
xavg3 <- c( mean( my_data[ my_data$y3 == 0, "x3" ] ), mean( my_data[ my_data$y3 == 1, "x3" ] ), 
mean( my_data[ my_data$y3 == 2, "x3" ] ),   mean( my_data[ my_data$y3 == 3, "x3" ] ),    mean( my_data[ my_data$y3 == 4, "x3" ] ), 
mean( my_data[ my_data$y3 == 5, "x3" ] ),   mean( my_data[ my_data$y3 == 6, "x3" ] ),    mean( my_data[ my_data$y3 == 7, "x3" ] ),
mean( my_data[ my_data$y3 == 8, "x3" ] ),   mean( my_data[ my_data$y3 == 9, "x3" ] ),    mean( my_data[ my_data$y3 == 10, "x3" ] ), 
mean( my_data[ my_data$y3 == 11, "x3" ] ), mean( my_data[ my_data$y3 == 12, "x3" ] ) , mean( my_data[ my_data$y3 == 13, "x3" ] ), 
mean( my_data[ my_data$y3 == 14, "x3" ] ), mean( my_data[ my_data$y3 == 15, "x3" ] ),  mean( my_data[ my_data$y3 == 16, "x3" ] ), 
mean( my_data[ my_data$y3 == 17, "x3" ] ), mean( my_data[ my_data$y3 == 18, "x3" ] ) )
plot( xavg3, yavg3 )


############################################################################################################

x4 <- ifelse ( ParcelData1 $DEM < 3451.614502 / 0.3048,  3451.614502/ 0.3048 - ParcelData1 $DEM, 0 )
y4 <- ParcelData1 $Cate4
plot(x4, y4)

my_data$x4 <- x4* 0.3048/1000
my_data$y4 <- y4


yavg4 <- c(0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
xavg4 <- c( mean( my_data[ my_data$y4 == 0, "x4" ] ), mean( my_data[ my_data$y4 == 1, "x4" ] ),  mean( my_data[ my_data$y4 == 2,   "x4" ] ), mean( my_data[ my_data$y4 == 3,   "x4" ] ), mean( my_data[ my_data$y4 == 4,   "x4" ] ), 		mean( my_data[ my_data$y4 == 5,   "x4" ] ),
mean( my_data[ my_data$y4 == 6,   "x4" ] ),  mean( my_data[ my_data$y4 == 7,   "x4" ] ), 		mean( my_data[ my_data$y4 == 8,   "x4" ] ), 
mean( my_data[ my_data$y4 == 9,   "x4" ] ),  mean( my_data[ my_data$y4 == 10, "x4" ] ), 		mean( my_data[ my_data$y4 == 11, "x4" ] ),
mean( my_data[ my_data$y4 == 12, "x4" ] ),  mean( my_data[ my_data$y4 == 13, "x4" ] ), 		mean( my_data[ my_data$y4 == 14, "x4" ] ),
mean( my_data[ my_data$y4 == 15, "x4" ] ),  mean( my_data[ my_data$y4 == 16, "x4" ] ), 		mean( my_data[ my_data$y4 == 17, "x4" ] ),
mean( my_data[ my_data$y4 == 18, "x4" ] ),  mean( my_data[ my_data$y4 == 19, "x4" ] ), 		mean( my_data[ my_data$y4 == 20, "x4" ] ),
mean( my_data[ my_data$y4 == 21, "x4" ] ) )
plot( xavg4, yavg4 )


############################################################################################################

x5 <- ifelse ( ParcelData1 $DEM < 3838.720651 / 0.3048,  3838.720651/ 0.3048 - ParcelData1 $DEM, 0 )
y5 <- ParcelData1 $Cate5
plot(x5, y5)

my_data$x5 <- x5* 0.3048/1000
my_data$y5 <- y5


yavg5 <- c( 0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 )
xavg5 <- c( mean( my_data[ my_data$y5 == 0, "x5" ] ), mean( my_data[ my_data$y5 == 1, "x5" ] ), 
mean( my_data[ my_data$y5 == 2,   "x5" ] ), mean( my_data[ my_data$y5 == 3,   "x5" ] ), 
mean( my_data[ my_data$y5 == 4,   "x5" ] ), mean( my_data[ my_data$y5 == 5,   "x5" ] ),
mean( my_data[ my_data$y5 == 6,   "x5" ] ), mean( my_data[ my_data$y5 == 7,   "x5" ] ),
mean( my_data[ my_data$y5 == 8,   "x5" ] ), mean( my_data[ my_data$y5 == 9,   "x5" ] ),
mean( my_data[ my_data$y5 == 10, "x5" ] ), mean( my_data[ my_data$y5 == 11, "x5" ] ),
mean( my_data[ my_data$y5 == 12, "x5" ] ), mean( my_data[ my_data$y5 == 13, "x5" ] ),
mean( my_data[ my_data$y5 == 14, "x5" ] ), mean( my_data[ my_data$y5 == 15, "x5" ] ),
mean( my_data[ my_data$y5 == 16, "x5" ] ), mean( my_data[ my_data$y5 == 17, "x5" ] ),
mean( my_data[ my_data$y5 == 18, "x5" ] ), mean( my_data[ my_data$y5 == 19, "x5" ] ),
mean( my_data[ my_data$y5 == 20, "x5" ] ), mean( my_data[ my_data$y5 == 21, "x5" ] ) )
plot( xavg5, yavg5 )

############################################################################################################


xy1 <- data.frame(x = xavg1, y = yavg1 * 0.3048 ) 
xy2 <- data.frame(x = xavg2, y = yavg2 * 0.3048)
xy3 <- data.frame(x = xavg3, y = yavg3 * 0.3048)
xy4 <- data.frame(x = xavg4, y = yavg4 * 0.3048)
xy5 <- data.frame(x = xavg5, y = yavg5 * 0.3048)


ggplot( data = xy1, aes(x = x, y = y) ,  color = "black") + theme_bw() + 
geom_point(data = xy2,  color = "red") +
geom_point(data = xy3, color = "blue") +
geom_point(data = xy4,  color = "green") +
geom_point(data = xy5, color = "orange") 


