library(ggplot2)

# import data
data = read.csv("total_center_angle.csv")
data$X = NULL

summary(data)
ggplot(data, aes(x = factor(frame_id), y = angle)) + geom_boxplot( )
ggplot(data, aes(x = angle)) + geom_density()
ggplot(data, aes(x = angle)) + geom_histogram(bins = 100)
