library(ggplot2)
library(dplyr)
library(reshape2)
# import data
data = read.csv("total_center_angle.csv")
nrow(data)
data$X = NULL

summary(data)
ggplot(data, aes(x = factor(frame_id), y = angle)) + geom_boxplot( )
ggplot(data, aes(x = angle)) + geom_density()
ggplot(data, aes(x = angle)) + geom_histogram(bins = 200)

original = read.csv("2nd_submission_daniel_kim_output.csv")
colnames(original) = c("frame_id", "original_steering_angle")
contrast = read.csv("2nd_submission_daniel_kim_contrast_output.csv")
colnames(contrast) = c("frame_id", "contrast_steering_angle")
data = left_join(original, contrast, c("frame_id" = "frame_id"))
data$avg_steering_angle = (data$original_steering_angle + data$contrast_steering_angle) / 2


data.melt = melt(data, id=c("frame_id"))
ggplot(data.melt, aes(color = factor(variable), x = value)) + geom_density()

data = arrange(data, frame_id)
ggplot(data.melt, aes(x = value, color =factor(variable))) + geom_histogram()
# ggplot()  + geom_line(data = data, aes(x = 1:nrow(data), data$steering_angle, color = 'red')) + geom_line(data = data, aes(x = 1:nrow(data), data$steering_angle, color = 'blue'))
ggplot(data = data.melt, aes(x = 1:nrow(data.melt), data$value)) + geom_line() + facet_grid(variable ~ .)
ggplot(data.melt, aes(x = value, fill =factor(variable))) + geom_histogram(position = "dodge")
ggplot(data = data)  + geom_line(aes(x = 1:nrow(data), data$original_steering_angle, color = 'red')) + geom_line(aes(x = 1:nrow(data), data$contrast_steering_angle, color = 'blue'))
ggplot(data = data)  + geom_line(aes(x = 1:nrow(data), data$original_steering_angle, color = 'red')) + geom_line(aes(x = 1:nrow(data), data$avg_steering_angle, color = 'blue'))

summary(data)
sd(data$original_steering_angle)
sd(data$contrast_steering_angle)
sd(data$avg_steering_angle)
