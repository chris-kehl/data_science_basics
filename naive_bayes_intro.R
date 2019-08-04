install.packages("e1071")
install.packages("gridExtra")
install.packages("taRifx")

library(e1071)
library(gridExtra)
library(tidyverse)
library(taRifx)

# load in the appropriate files
wine_tr <- read_csv("Documents/data_files/data_science/data_sets/website_data_sets/wine_flag_training.csv")

wine_test <- read_csv("Documents/data_files/data_science/data_sets/website_data_sets/wine_flag_test.csv")

wine_test <- na.omit(wine_test)

# build 1st of two contengency tables showing the Type(red, white) and Alcohol flag
ta <- table(wine_tr$Type, wine_tr$Alcohol_flag)
colnames(ta) <- c("Alcohol = High", "Alcohol = Low")
rownames(ta) <- c("Type = Red", "Type = White")
addmargins(A = ta, FUN = list(Total = sum), quiet = TRUE)

# build the second contengency table Type(red, white) and Sugar flag
ts <- table(wine_tr$Type, wine_tr$Sugar_flag)
colnames(ts) <- c("Sugar = High", "Sugar = Low")
rownames(ts) <- c("Type = Red", "Type = White")
addmargins(A = ts, FUN = list(Total = sum), quiet = TRUE)

# Visualize the data with bar graphs
plot1 <- ggplot(wine_tr, aes(Type)) + geom_bar( aes(fill = Alcohol_flag), position = "fill") +
  ylab("Proportion")
plot2 <- ggplot(wine_tr, aes(Type)) + geom_bar( aes(fill = Sugar_flag), position = "fill") +
  ylab("Proportion")
grid.arrange(plot1, plot2, nrow = 1)

# run the Naive Bayes algorithm
nb01 <- naiveBayes(formula = Type ~ Alcohol_flag + Sugar_flag, data = wine_tr)

nb01

# predict the type of wine in our test data set
#  NOTE: I'm getting an error "All arguments must have same length"
wine_test <- na.omit(wine_test)
ypred <- predict(object = nb01, newdata = wine_test)

# added in atempt to remove NA values
ypred <- na.omit(ypred)
ypred <- na.omit(data.matrix)
# create the contengency tables of actual vs predicted
t.preds <- table(wine_test$Type, ypred)
rownames(t.preds) <- c("Actual: Red", "Actual: White")
colnames(t.preds) <- c("Predicted: Red", "Predicted: White")
addmargins(A = t.preds, FUN = list(Total = sum), quiet = TRUE)











