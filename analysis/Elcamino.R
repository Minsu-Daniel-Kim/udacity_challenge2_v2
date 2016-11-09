library(data.table)
library(dplyr)
library(arules)
library(ggplot2)
data1 <- fread("interpolated_1.csv")
data2$file <- 'data1'
data2 <- fread("interpolated-2.csv")
data2$file <- 'data2'
data3 <- fread("interpolated-3.csv")
data3$file <- 'data3'
data4 <- fread("interpolated-4.csv")
data4$file <- 'data4'
data5 <- fread("interpolated-5.csv")
data5$file <- 'data5'
data6 <- fread("interpolated-6.csv")
data6$file <- 'data6'


data <- rbind(data2, data3, data4, data5, data6)

data.center <- filter(data, frame_id == 'center_camera')

data.center <- select(data.center, filename, angle, speed)

data.center %>% ggplot(aes(x = speed, y = angle)) + geom_point() + xlim(c(5, 20))


quantile(data.center$speed, probs = c(0, 0.01, 0.1, 0.9, 1))


tmp <- filter(data.center, speed < 5) %>% select(angle)
var(tmp$angle)
tmp %>% ggplot(aes(x = angle)) + geom_histogram(bins=100)
tmp2 <- filter(data.center, speed >= 5, speed < 10) %>% select(angle)
var(tmp2$angle)
tmp2 %>% ggplot(aes(x = angle)) + geom_histogram(bins=100)
tmp3 <- filter(data.center, speed >= 10, speed < 15) %>% select(angle)
var(tmp3$angle)

summary(tmp2)



ggplot(data.center, aes(x =angle)) + geom_histogram(bins = 1000) + xlim(c(-1, 1))

discretize(data.center$angle, method = 'frequency', categories = 5)
data.center$label = discretize(data.center$angle, method = 'frequency', categories = 5)

head(data.center)


ggplot(data.center, aes(x =angle, fill = label)) + geom_histogram(bins = 300) + xlim(c(-1, 1))

filter(data.center, label == 1) %>% select(angle) %>% summary()

ggplot(filter(data.center, label == 5), aes(x = angle)) + geom_histogram(bins = 200)

filter(data.center, speed !=0, -2 < angle, angle < 2) %>% 
  ggplot(aes(x = angle)) + geom_histogram(bins = 500)
tmp <- filter(data.center, speed !=0, -2 < angle, angle < 2) %>% select(angle)
sqrt(mean(tmp$angle ^ 2))
sqrt(mean(( tmp$angle - mean(tmp$angle) ) ^ 2))

filter(data.center, filename == 'center/1475187636603494082.png') %>% select(angle)

# 1. 전체 angle 분포 & 전체 speed 분포
# 2. 전체 angle vs 전체 speed
# 3. 정차, 직선, 좌우 threshold

# 1. 전체 angle 분포 & 전체 speed 분포
ggplot(data.center, aes(x = angle)) + geom_histogram(bins = 100)
ggplot(data.center, aes(x = angle)) + geom_histogram(bins = 100) + xlim(c(-0.5, 0.5))

# 2. 전체 angle vs 전체 speed
ggplot(data.center, aes(x = speed, y = angle)) + geom_point()

# 3. speed가 0인경우의 분포
data.center.zero_speed <- filter(data.center, speed == 0)

filter(data.center.zero_speed, angle > 1 | angle < -1, file == 'data4') %>% View

data.center.zero_speed %>% nrow
ggplot(data.center.zero_speed, aes(x = angle)) + geom_histogram(bins = 200) + xlim(c(-0.5, 0.5))


## speed가 0 이면서 angle 값이 큰경우
filter(data.center.zero_speed, angle > 0.2 | angle < -0.2) %>% nrow / nrow(data.center.zero_speed)
quantile(data.center.zero_speed$angle)
ggplot(data.center.zero_speed, aes(x = angle)) + geom_histogram(bins = 200)

filter(data.center, speed > -0.2, speed < 0.2, speed !=0) %>% select(angle) %>% sum

## almost speed zero
ggplot(filter(data.center, speed > -0.2, speed < 0.2), aes(x = speed, y = angle)) + geom_point()
ggplot(filter(data.center, speed > -0.2, speed < 0.2, speed !=0), aes(x = speed, y = angle)) + geom_point()

# torque
ggplot(data.center, aes(x = torque)) + geom_histogram()

ggplot(data.center, aes(x = torque, y = speed)) + geom_histogram()


filter(data.center, speed > -0.2, speed < 0.2, speed !=0) %>% filter(angle < -1 | angle > 1)
filter(data.center, speed == 0, -0.3 < angle, angle < 0.3, file == 'data5') %>% head(100)

filter(data.center, angle ==0, speed == 0) %>% nrow / nrow(filter(data.center, speed == 0))

ang <- filter(data.center, speed > -0.2, speed < 0.2, speed !=0, angle < 1, angle > -1)
ggplot(filter(data.center, speed > -0.2, speed < 0.2, speed !=0), aes(x = angle)) + geom_histogram()
sqrt(mean(ang$angle ^ 2))

# finding
# 1. 쓰레기는 speed가 영이면서 앵글 값이 큰경우. -0.25 > angle , angle > 1
filter(data.center, speed == 0, angle < - 1. | angle > 1) %>% nrow
# 2. 정차는 -0.2 < speed < 0.2
filter(data.center, -0.2 < speed, speed < 0.2, frame_id == 'center_camera') %>%
  ggplot(aes(x = angle)) + geom_histogram() + xlim(c(-0.2, 0.2))
filter(data.center, -0.2 < speed, speed < 0.2, -0.01 < angle, angle < 0.01, frame_id == 'center_camera') %>% nrow
  ggplot(aes(x = angle)) + geom_histogram()
filter(data.center, -0.2 < speed, speed < 0.2, -1 < angle, angle < 1, frame_id == 'center_camera', file=='data6') %>%
  select(filename) %>% tail(300)

tmp <- filter(data.center, -0.2 < speed, speed < 0.2, -1 < angle, angle < 1, frame_id == 'center_camera', file=='data6')
tmp_diff <- setdiff(data.center$timestamp, tmp$timestamp)

tmp_contrast <- data.center[data.center$timestamp %in% tmp_diff,]

c(tmp$angle, tmp_contrast$angle)
sqrt(mean(c(tmp$angle, tmp_contrast$angle) ^ 2))


sqrt(mean((data.center$angle - mean(tmp$angle))^ 2))

sqrt(mean(tmp ^ 2)) + 
tmp2 <- filter(data.center, -0.2 > speed | speed > 0.2 | -1 > angle | angle > 1, frame_id == 'center_camera', file != 'data6') %>%
  select(angle)

sqrt(mean((tmp - rnorm(nrow(tmp), 2)) ^ 2))

summary(data.center.zero_speed$angle)
quantile(data.center.zero_speed$angle, probs = c(0, 0.1, 0.9, 1))
?quantile





ggplot(data, aes(x = speed, y = angle)) + geom_point()

# speed distribution
filter(data, speed < 1) %>% ggplot(aes(x = speed)) + geom_histogram(bins = 100)

filter(data, speed < 1) %>% nrow()

filter(data, speed == 0) %>% head()
filter(data, speed < 1) %>% ggplot(aes(x = angle)) + geom_histogram(bins = 1000)

filter(data, speed < 0.1) %>% select(angle) %>% summary()
filter(data, angle < -1, speed < 1)

filter(data, speed > 2) %>% ggplot(aes(x = angle)) + geom_histogram(bins = 1000)



