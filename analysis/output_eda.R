library(ggplot2)
library(dplyr)
library(reshape2)

original <- read.csv("original_output.csv")
contrast <- read.csv("original_contrast_output.csv")
data <- cbind(original, contrast$steering_angle)
colnames(data) <- c('frame_id', 'original_steering_angle', 'contrast_steering_angle')
head(data)
data.melt = melt(data, id=c("frame_id"))

get_mse <- function(pred, real) {
  
  mean((pred - real) ^ 2)
  
}

get_corrected_value <- function(val) {
  
  threshold = sd(val) * 0.01 + mean(val)
  return(threshold)
  
  
}

data$avg_steering_angle = (data$original_steering_angle + data$contrast_steering_angle) / 2

# histogram

ggplot(data= data, aes(x = frame_id)) + geom_histogram(bins = 200)

arrange(data, frame_id) %>%
  ggplot() + geom_line(aes(x = 1:nrow(data), y = frame_id), colour = 'blue', size = 4)  +
  geom_line(aes(x = 1:nrow(data), y = original_steering_angle), colour = 'red', size = 2) +
  geom_line(aes(x = 1:nrow(data), y = contrast_steering_angle), colour = 'yellow')

arrange(data, frame_id) %>%
  ggplot() +
  geom_line(aes(x = 1:nrow(data), y = original_steering_angle), colour = 'red') 
arrange(data, frame_id) %>%
  ggplot() +
  geom_line(aes(x = 1:nrow(data), y = contrast_steering_angle), colour = 'blue')

arrange(data, frame_id) %>%
  ggplot() +
  geom_line(aes(x = 1:nrow(data), y = avg_steering_angle), colour = 'blue')

# direction
data$to_right <- data$frame_id > 0
table(data$to_right)

# the same direction
sum(sign(data$frame_id) == sign(data$original_steering_angle)) / nrow(data)
# assign true/false to same direction
data$same_direction = sign(data$frame_id) == sign(data$original_steering_angle)
# correct incorrect steering to 0 
data$original_steering_angle_corrected = data$original_steering_angle
# data$original_steering_angle_corrected2 = data$original_steering_angle
data$original_steering_angle_corrected[!data$same_direction] = 0
# data$original_steering_angle_corrected2[!data$same_direction] = data$original_steering_angle_correctedw[!data$same_direction] * -1
# plot again
arrange(data, frame_id) %>%
  ggplot() + geom_line(aes(x = 1:nrow(data), y = frame_id), colour = 'blue', size = 2)  +
  geom_line(aes(x = 1:nrow(data), y = original_steering_angle), colour = 'red', size =1)



# data$original_steering_angle_corrected 
# data$same_direction
# filter(data, !same_direction)

thred = get_corrected_value(data$original_steering_angle)
data$original_steering_angle_corrected2 <- data$original_steering_angle
data$original_steering_angle_corrected2[data$original_steering_angle < thred & data$original_steering_angle > thred * -1] = 0

get_mse(data$frame_id, data$original_steering_angle)
get_mse(data$frame_id, data$original_steering_angle_corrected)
get_mse(data$frame_id, data$original_steering_angle_corrected2)
