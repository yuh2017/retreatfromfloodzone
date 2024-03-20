pth1 = "./traing_scores.csv"
#pth0 = "C:/Users/zanwa/OneDrive/Desktop/FLCounty/traing_scores.csv"
MyDatax1 <- read.csv(file= pth1, header=TRUE)
colnames(MyDatax1) <- c("learning_rates", "50", "100", "200", "400", "600", "800", "1000")

long <- melt( MyDatax1, id.vars = c("learning_rates"), variable.name = "n_estimators")

mine.heatmap <- ggplot(data = long, mapping = aes(x = variable,
					y = learning_rates, fill = sqrt(value) )) +
                       geom_tile() + xlab(label = "Sample") +
                scale_fill_gradient(name = "Testing accuracy \n 2015 land use", 
                low = "#FFFFFF", high = "#012345") +
                xlab(label = "Number of estimators") +
			  ylab(label = "Learning rates") +
                theme(strip.placement = "outside")



png(file= "./test_accuracy.png", ,width= 18,height= 12,units="cm",res=300)
mine.heatmap
dev.off()