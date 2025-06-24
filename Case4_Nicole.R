setwd("/Users/wqx/Desktop/Machine Learning/HW/HW4")

rws.5k=read.csv("yelp.biz.sample.csv")
colnames(rws.5k)

#1.1
glm.model1 <- glm(is_open ~ elite_cnt + price_level, data = rws.5k, family = "binomial")
summary(glm.model1)
exp(glm.model1$coefficients[2:5])

#1.2
pred_probs <- predict(glm.model1, type = "response")

library(pROC)
roc_obj <- roc(rws.5k$is_open, pred_probs)

plot(roc_obj, main = "ROC Curve", print.auc = T, 
     legacy.axes = TRUE, lwd = 2)

cost_FP <- 55  # cost per false positive
cost_FN <- 20  # cost per false negative

library(dplyr)
costs <- coords(roc_obj, "all", 
                ret = c("threshold", "fp", "fn")) %>%
  mutate(total_cost = fp * cost_FP + fn * cost_FN)

# Find optimal threshold: minimizing total cost
optimal_threshold <- costs$threshold[which.min(costs$total_cost)]
print(paste("Optimal threshold:", round(optimal_threshold, 4)))


#1.3    
youden <- coords(roc_obj, "all", ret = c("threshold", "sensitivity", "specificity"))
youden <- youden %>%
  mutate(youden_index = sensitivity + specificity - 1) %>%
  arrange(desc(youden_index))

youden$youden_j <- youden$sensitivity + youden$specificity - 1
optimal_idx <- which.max(youden$youden_j)
optimal_threshold2 <- youden$threshold[optimal_idx]
print(paste("Optimal Threshold (Youden's Index):", round(optimal_threshold2, 3)))
#The optimal threshold is 0.574

predicted_labels <- ifelse(pred_probs > optimal_threshold2, 1, 0)
confusion_matrix <- table(Predicted = predicted_labels, Actual = rws.5k$is_open)
print(confusion_matrix)


#1.4
accuracy <- coords(roc_obj, "all", ret = c("threshold", "accuracy")) %>%
  filter (threshold <= optimal_threshold & threshold >= optimal_threshold2) %>%
  mutate(mean_a = mean(accuracy))
#The mean accuracy is 0.5751347


#1.5
glm.model2 = glm(is_open ~ elite_cnt + price_level * biz.stars + repeated_cnt, data=rws.5k, family =
                binomial)
summary(glm.model2)

predictions1 <- predict(glm.model1, type = "response")
predictions2 <- predict(glm.model2, type = "response")

## ROCs of different models
auc1 <- auc(roc(rws.5k$is_open, predictions1))
auc2 <- auc(roc(rws.5k$is_open, predictions2))

print(auc1) #0.646
print(auc2) #0.703


#1.6

rws.5k$pred_prob <- predict(glm.model2, type = "response")  # Predicted probabilities

library(gains)
gains_table <- gains(actual = rws.5k$is_open, predicted = rws.5k$pred_prob, 
                     groups = 10)
print(gains_table)
plot(gains_table)
###By targeting 50% of the restaurants, the platform can already capture 60.5% of the actual open restaurants.


#2.1.1
set.seed(123)
train_indices <- sample(1:nrow(rws.5k), 
                        size = 0.7 * nrow(rws.5k))
train_data <- rws.5k[train_indices, ]
test_data <- rws.5k[-train_indices, ]

glm.model1 <- glm(is_open ~ elite_cnt + price_level, data = rws.5k, family = "binomial")
glm.model2 = glm(is_open ~ elite_cnt + price_level * biz.stars + repeated_cnt, data=rws.5k, family =
                   binomial)

train_pred1 <- predict(glm.model1, train_data, type = "response")
train_pred2 <- predict(glm.model2, train_data, type = "response")

test_pred1 <- predict(glm.model1, test_data, type = "response")
test_pred2 <- predict(glm.model2, test_data, type = "response")

train_auc <- c(
  auc(roc(train_data$is_open, train_pred1)),
  auc(roc(train_data$is_open, train_pred2))
)
names(train_auc) <- c("Model 1", "Model 2")

test_auc <- c(
  auc(roc(test_data$is_open, test_pred1)),
  auc(roc(test_data$is_open, test_pred2))
)
names(test_auc) <- c("Model 1", "Model 2")
print(train_auc)
print(test_auc)

#2.2
glm.model3 <- glm(is_open ~ poly(elite_cnt, 2, raw=T) + price_level*biz.stars*repeated_cnt + city,
    rws.5k, family=binomial)


set.seed(123)  
data <- rws.5k  
k <- 10

folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)

train_auc1 <- numeric(k)
train_auc2 <- numeric(k)
train_auc3 <- numeric(k)
test_auc1 <- numeric(k)
test_auc2 <- numeric(k)
test_auc3 <- numeric(k)

for(i in 1:k) {
  test_indices <- which(folds == i)
  train_data <- data[-test_indices, ]
  test_data <- data[test_indices, ]
  
  glm.model1 <- glm(is_open ~ elite_cnt + price_level, data = train_data, family = "binomial")
  glm.model2 <- glm(is_open ~ elite_cnt + price_level * biz.stars + repeated_cnt, data=train_data, family =
                     binomial)
  glm.model3 <- glm(is_open ~ poly(elite_cnt, 2, raw=T) + price_level*biz.stars*repeated_cnt + city,
                    train_data, family=binomial)
  
  train_pred1 <- predict(glm.model1, train_data, type = "response")
  train_pred2 <- predict(glm.model2, train_data, type = "response")
  train_pred3 <- predict(glm.model3, train_data, type = "response")
  
  test_pred1 <- predict(glm.model1, test_data, type = "response")
  test_pred2 <- predict(glm.model2, test_data, type = "response")
  test_pred3 <- predict(glm.model3, test_data, type = "response")
  
    train_auc1[i] <-auc(roc(train_data$is_open, train_pred1))
    train_auc2[i] <-auc(roc(train_data$is_open, train_pred2))
    train_auc3[i] <-auc(roc(train_data$is_open, train_pred3))

    test_auc1[i] <- auc(roc(test_data$is_open, test_pred1))
    test_auc2[i] <- auc(roc(test_data$is_open, test_pred2))
    test_auc3[i] <- auc(roc(test_data$is_open, test_pred3))
}


cv_results <- data.frame(
  Fold = 1:10,
  Train_auc1 = train_auc1,
  Train_auc2 = train_auc2,
  Train_auc3 = train_auc3,
  Test_auc1 = test_auc1,
  Test_auc2 = test_auc2,
  Test_auc3 = test_auc3
)

print(cv_results)
mean_auc <- data.frame(
  Metric = c("Train_auc1", "Train_auc2", "Train_auc3", 
             "Test_auc1", "Test_auc2", "Test_auc3"),
  Mean_AUC = c(mean(train_auc1),
               mean(train_auc2),
               mean(train_auc3),
               mean(test_auc1),
               mean(test_auc2),
               mean(test_auc3))
)

print(mean_auc)

#      Metric  Mean_AUC
# 1 Train_auc1 0.6460041
# 2 Train_auc2 0.7031238
# 3 Train_auc3 0.7031238
# 4  Test_auc1 0.6457346
# 5  Test_auc2 0.7002970
# 6  Test_auc3 0.7130685




