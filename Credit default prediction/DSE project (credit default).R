
#Packages 

library(ISLR) # contains the dataset "Default"
library(ROCR) # for constructing ROC curves
library(kknn)# allows us to do KNN for regression and classification
library(e1071) ##NB classifier 
library(ggplot2)

#Read and load excel file 

rm(list = ls()) #clear
library(readxl)
default_data = read_excel("Credit card default subsample.xls", col_names = TRUE) #read data
default_data <- default_data[rowSums(!is.na(default_data)) > 0, ] #only remove rows that are entirely empty, keep the rest of data
rownames(default_data) <- NULL #resets row names for tidiness 

default_data = default_data[-1, ] #remove headers

sum(is.na(default_data)) #check if theres NA values

default_data$X1 = as.numeric(default_data$X1) #change str to number

default_data$X2 = factor(default_data$X2, levels = c(1, 2), labels = c("Male", "Female")) 

default_data$X3 = factor(default_data$X3, levels = c(1, 2, 3, 4, 5, 6), 
                 labels = c("Graduate School", "University", "High School", "Other", 
                            "Data unavailable", "Data unavailable"))

default_data$X3[is.na(default_data$X3)] <- "Data unavailable" #For datas that are not 1-6, replace with "Data unavailable"

default_data$X4 = factor(default_data$X4, levels = c(0, 1, 2, 3), labels = c("Missing", "Married", "Single", "Other"))

default_data$X4[is.na(default_data$X4)] <- "Missing" #For datas that are not 0-3, replace with "Missing"

default_data$X5 = as.numeric(default_data$X5) #change str to number

#X6-11: change str to number
for (v in c("X6","X7","X8","X9","X10","X11")) default_data[[v]] <- as.integer(default_data[[v]])

#X12-X23 : change str to number 
for (v in paste0("X",12:23)) default_data[[v]] <- as.numeric(as.character(default_data[[v]]))

#default outcome (No = 0, Yes = 1)
default_data$Y = factor(default_data$Y, levels = c(0, 1), labels = c("No", "Yes"))


set.seed(8867)
ntrain = floor(0.7 * nrow(default_data)) #70/30 train-test split 
tr = sample(seq_len(nrow(default_data)),ntrain) # draw ntrain observations from original data
data_train = default_data[tr,] # training sample
data_test = default_data[-tr,] # testing sample


#As credit limit (X1), age (X5), bill amounts (X12-17), and previous payments (X18-X23)
#tends to be highly right-skewed, we perform log transformation to compresses large values. 

#Log transformation: 

#Log X1 (credit limit)

data_train$log_X1 = data_train$X1 #create a working copy so original data stay intact 
data_train$log_X1[data_train$log_X1 < 0] <- 0  #replaces negatives with 0 (or a tiny constant)
data_train$log_X1 <- log1p(data_train$log_X1) #computes log(1 + p), safely handles zero as log(0) is undefined. 

#Log X5 (age)

data_train$log_X5 <- log1p(data_train$X5)

#Log X18-X23 (previous payments)
clamp_nonneg <- function(x){ x[is.na(x)] <- 0; x[x < 0] <- 0; x } #functon to clean negatives 
for (v in c("X18","X19","X20","X21","X22","X23")) {
  data_train[[v]] <- clamp_nonneg(data_train[[v]])
} #clean any negatives with "clamp_nonneg"

data_train$log_X18 <- log1p(data_train$X18)
data_train$log_X19 <- log1p(data_train$X19)
data_train$log_X20 <- log1p(data_train$X20)
data_train$log_X21 <- log1p(data_train$X21)
data_train$log_X22 <- log1p(data_train$X22)
data_train$log_X23 <- log1p(data_train$X23)

#since the amount paid from April to September can vary widely (some clients pay just the minimum; others clear large balances)
#taking logs reduces the effect of huge payment values so that model coefficients are not driven by outliers. 

#Apply log transformation to test data:

data_test$log_X1 = data_test$X1 #create a working copy so original data stay intact 
data_test$log_X1[data_test$log_X1 < 0] <- 0  #replaces negatives with 0 (or a tiny constant)
data_test$log_X1 <- log1p(data_test$log_X1) 

data_test$log_X5 <- log1p(data_test$X5)

for (v in c("X18","X19","X20","X21","X22","X23")) {
  data_test[[v]] <- clamp_nonneg(data_test[[v]])
} #clean any negatives with "clamp_nonneg" 

data_test$log_X18 <- log1p(data_test$X18)
data_test$log_X19 <- log1p(data_test$X19)
data_test$log_X20 <- log1p(data_test$X20)
data_test$log_X21 <- log1p(data_test$X21)
data_test$log_X22 <- log1p(data_test$X22)
data_test$log_X23 <- log1p(data_test$X23)

#Collapse X12-X17 (six monthly bill statement amounts) into a single summary statistic (median) per person: 

#chose median (instead of mean) as a more robust measurement since bills are right-skewed. 

data_train$median_bill = apply(data_train[, c("X12", "X13", "X14", "X15", "X16", "X17")], 1, median, na.rm = TRUE)
data_train$median_bill[data_train$median_bill < 0] <- 0 #clean any negatives
data_train$log_med_bill <- log1p(data_train$median_bill)

data_test$median_bill <- apply(data_test[, c("X12","X13","X14","X15","X16","X17")],
                               1, median, na.rm = TRUE)
data_test$median_bill[data_test$median_bill < 0] <- 0
data_test$log_med_bill <- log1p(data_test$median_bill)

##########################################################
########## DESCRIPTIVE STATS##########
##########################################################
library(readxl)
###SUMMARY OF DATA###
head(default_data)          # first 6 rows
summary(default_data)       # min, median, mean, NA count, etc.
nrow(default_data)  # total number of customers
ncol(default_data)  # total number of variables

###FREQEUENCY TABLES FOR CATEGORICAL VARIABLES###
cat_vars <- c("X2","X3","X4","Y")
cat_summary <- lapply(cat_vars, function(v){
  tb <- table(default_data[[v]], useNA = "ifany")
  prop <- prop.table(tb)
  data.frame(Variable = v, Level = names(tb),
             Count = as.integer(tb), Prop = as.numeric(round(prop, 3)))
})
cat_summary <- do.call(rbind, cat_summary)
cat_summary

cat_compare <- function(v){
  tb <- prop.table(table(default_data[[v]], default_data$Y), margin = 2)  # normalize by Y
  out <- as.data.frame.matrix(tb)
  out$Level <- rownames(out)
  out$Variable <- v
  rownames(out) <- NULL
  out
}

cat_byY <- rbind(
  cat_compare("X2"),  # Gender
  cat_compare("X3"),  # Education
  cat_compare("X4")   # Marital status
)

cat_byY

library(ggplot2)

ggplot(cat_byY, aes(x = Level, y = Yes, fill = Level)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ Variable, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(title = "Proportion of Defaulters Within Each Category",
       x = "Category Level",
       y = "Default Rate (%)",
       fill = "Category Level") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

#install.packages("tidyverse")
library(dplyr)
library(tidyr)
library(scales)
#select the relevant columns using explicit dplyr::select()
data_long <- dplyr::select(data_train, Y, X6:X11) %>%
  pivot_longer(cols = X6:X11, 
               names_to = "Month", 
               values_to = "Repayment_Status") %>%  # reshape data to long format
  mutate(Month = factor(Month, 
                        levels = c("X6", "X7", "X8", "X9", "X10", "X11"),
                        labels = c("Sep 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005")))

#create a bar plot showing proportions of defaulters and non-defaulters for each repayment status by month
ggplot(data_long, aes(x = Repayment_Status, fill = Y)) +
  geom_bar(position = "fill") +   # show proportions
  facet_wrap(~ Month, scales = "free_y") +
  labs(title = "Proportion of Default by Repayment Status (X6 - X11)", 
       x = "Repayment Status", 
       y = "Proportion", 
       fill = "Default Payment Status") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

####################################
##### DENSITY OF CREDIT AMOUNT #####
####################################
ggplot(data_train, aes(x = X1, fill = as.factor(Y))) +
  geom_density(alpha = 0.5, adjust = 1.5, color = "black") +
  scale_fill_manual(values = c("darkolivegreen3", "coral1")) +  # Color for non-defaulters and defaulters
  labs(
    title = "Density Plot of Credit Amount by Default Payment Status",
    x = "Credit Amount (NT Dollars)",
    y = "Density",
    fill = "Default Payment Status"
  ) +
  theme_minimal() +
  scale_x_continuous(
    labels = scales::label_number(scale = 1e-3, suffix = "K"),  # Scale down to thousands and add "K"
    breaks = seq(-100000, 600000, by = 75000),  # Breaks for the x-axis, adjust the "by" value
    limits = c(-100000, 600000)  # Sets the x-axis limits
  )

############################
###### DENSITY OF AGE ######
############################
#Density plot: AGE for each Default Status
ggplot(data_train, aes(x = X5, fill = as.factor(Y))) +
  geom_density(alpha = 0.5, adjust = 1.5, color = "black") +
  scale_fill_manual(values = c("darkolivegreen3", "coral1")) +  # Color for non-defaulters and defaulters
  labs(
    title = "Density Plot of Age by Default Payment Status",
    x = "Age",
    y = "Density",
    fill = "Default Payment Status"
  ) +
  theme_minimal() +
  scale_x_continuous(
    breaks = seq(0, 100, by = 5),  # Set more frequent breaks
    limits = c(10, 75)  # Set the x-axis limits in a single call
  )

###################################################
###### DENSITY OF INDIV PAYMENTS (X18 - X23) ######
###################################################

#Density plot: payment amount for each Default Status (X18 as example)
ggplot(data_train, aes(x = X18, fill = as.factor(Y))) +
  geom_density(alpha = 0.5, adjust = 1.5, color = "black") +
  scale_fill_manual(values = c("darkolivegreen3", "coral1")) +  # Color for non-defaulters and defaulters
  labs(
    title = "Density Plot of X18 by Default Payment Status",
    x = "Prev Payment Amt [X18] (NT Dollars)",
    y = "Density",
    fill = "Default Payment Status"
  ) +
  theme_minimal() +
  scale_x_continuous(
    labels = scales::label_number(scale = 1e-3, suffix = "K"),
    breaks = seq(-15000, 100000, by = 5000),  
    limits = c(-5000, 30000)  
  )

##########################################################
################### DATA VISUALISATION ###################
##########################################################
#install.packages("ggplot2")
#install.packages("dylyr")
library(ggplot2)
library(dplyr)

###Selecting more important factors influencing Default: 

###1. Credit Limit (X1) VS Default 
#shows how much credit the bank extended to the customer 

#credit limit histogram
ggplot(data_train, aes(x = log_X1, fill = Y)) +
  geom_histogram(alpha = 0.7, bins = 40, position = "identity") +
  labs(
    title = "Distribution of Credit Limit (log scale)",
    subtitle = "Defaulters tend to have slightly lower credit limits",
    x = "log(Credit Limit)",
    y = "Count",
    fill = "Default"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 16, hjust = 0.5)
  )

#histogram compares the distribution of credit limits between defaulters and non-defaulters



##########################################################
########## CORRELATIONS OF CONTINUOUS VARIABLES ##########
##########################################################
#install.packages("ggplot2")
library(ggplot2)
#use correlation heatmap to check for multicollinearity 

# Select only continuous variables
continuous_vars = dplyr::select(data_train, X1, X5, X12:X17, X18:X23)
# Calculate the correlation matrix
cor_matrix = cor(continuous_vars, use = "complete.obs", method = "pearson")
cor_melted = as.data.frame(as.table(cor_matrix))  # Convert matrix to data frame

ggplot(cor_melted, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limits = c(-1, 1)) +
  theme_minimal() +
  labs(title = "Correlation Heatmap of Continuous Variables",
       x = "Variable",
       y = "Variable",
       fill = "Correlation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Observation: X12 - X17 are highly correlated, need to create new aggregate variable!


########################################################
#### CORRELATION HEATMAP AFTER CREATING LOG_MED_BILL####
########################################################
continuous_vars = dplyr::select(data_train, X1, X5, X12:X17, log_med_bill, X18:X23)

# Calculate the correlation matrix
cor_matrix = cor(continuous_vars, use = "complete.obs", method = "pearson")
cor_melted = as.data.frame(as.table(cor_matrix))  # Convert matrix to data frame

ggplot(cor_melted, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limits = c(-1, 1)) +
  theme_minimal() +
  labs(title = "Correlation Heatmap of Continuous Variables",
       x = "Variable",
       y = "Variable",
       fill = "Correlation") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#multicollinearity issues solved!

#Variable selection (Forward selection): 
library(MASS)
null_model <- glm(Y ~ 1, data = data_train, family = binomial)
full_model <- glm(Y ~ log_X1 + X2 + X3 + X4 + log_X5 + X6 + X7 + X8 + X9 + X10 + X11 + 
                    log_med_bill + log_X18 + log_X19 + log_X20 + log_X21 + log_X22 + log_X23,
                  data = data_train, family = binomial)

#Forward selection by AIC 
fwd_model <- stepAIC(null_model, scope = list(lower = null_model, upper = formula(full_model)),
                     direction = "forward", trace = TRUE) 

summary(fwd_model)
attr(terms(fwd_model), "term.labels")

#Selected predictors (11 predictors): 
# "X6"           "log_X1"       "X9"           "log_X19"      "log_X18"      "X11"         
# "X3"           "log_X21"      "log_X5"       "log_X20"      "log_med_bill"

#######################################
####Logit model training and prediction
#######################################

#Fit logit model on training data with selected predictors:

glm_fit = glm(Y ~ X6 + log_X1 + X9 + log_X19 + log_X18 + X11 + X3 + log_X21 + log_X5 + log_X20 + log_med_bill, 
              data = data_train, family = "binomial")

summary(glm_fit)

#Predict the test observations using the trained logit model:
glm_prob = predict(glm_fit, newdata = data_test, type = "response")

#Build the confusion matrix:
table(glm_prob > 0.5, data_test$Y)

#Compute and plot the ROC curve:
pred = prediction(glm_prob, data_test$Y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2, main="ROC for Logit Model") # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2))) # Compute AUC and add text to ROC plot.
#########################################
#ROC=0.75
#########################################

#2. KNN 

#Scaling 

num_knn <- c("log_X1","log_X19","log_X18","log_X21","log_X5","log_X20","log_med_bill") #predictors to scale

mu  <- sapply(data_train[, num_knn], mean) #calculating mean of each predictor
sig <- sapply(data_train[, num_knn], sd) #calculating sd of each predictor
scale_cols <- function(df){
  out <- df
  for (v in num_knn) out[[v]] <- (out[[v]] - mu[[v]]) / sig[[v]]
  out
} #this function does z-score normalisation 

data_train_sc <- scale_cols(data_train) #now the training data is scaled 
data_test_sc  <- scale_cols(data_test) #test data is scaled using the same mu and sd from training

#loocv 
df.loocv = train.kknn(Y ~ log_X1 + log_X19 + log_X18 + log_X21 + log_X5 + log_X20 + log_med_bill,  
                      data = data_train_sc, kmax=100, kernel = "rectangular") #categorical variables removed

plot((1:100), df.loocv$MISCLASS, type = "l", col = "blue", 
     main = "LOOCV Misclassification", xlab = "Complexity: K", ylab = "Misclassification rate")

kbest = df.loocv$best.parameters$k
kbest
#kbest is 24
abline(v = kbest, col = "red", lty = 2, lwd = 2) 

text(x = kbest * 1.4,
     y = max(df.loocv$MISCLASS), 
     labels = paste("kbest =", kbest),
     col = "red", font = 2)


knnpredcv = kknn(Y ~  log_X1 + log_X19 + log_X18 + log_X21 + log_X5 + log_X20 + log_med_bill, data_train_sc, data_test_sc, k = kbest, kernel = "rectangular")

###### JUDGING MODEL PERFORMANCE ######
#confusion matrix
table(knnpredcv$fitted.values, data_test$Y)

#ROC
pred = prediction(knnpredcv$prob[,2], data_test$Y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC
plot(perf, col = "steelblue", lwd = 2, main="ROC for KNN", cex.main = 1.8,
     cex.lab = 1.5) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2)), cex = 2) # Compute AUC and add text to ROC plot.

#KNN AUC is 0.7

#3. Naive Bayes 
nbfit = naiveBayes(Y ~ log_X1 + X2 + X3 + X4 + log_X5 + X6 + X7 + X8 + X9 + X10 + X11 + 
                     log_med_bill + log_X18 + log_X19 + log_X20 + log_X21 + log_X22 + log_X23, 
                   data = data_train) 
#test on test set
nbpred = predict(nbfit, data_test, type = "class")
summary(nbpred)
# PREDICTION: of the 2100 obs in test set, 1500 did not default, while 600 defaulted

#get predicted probabilities of default for given X var:
nbpred2 = predict(nbfit, data_test, type = "raw")
summary(nbpred2) #[No] Mean = 0.65 // [Yes] Mean = 0.35

###### JUDGING MODEL PERFORMANCE ######
#confusion matrix:
cm_nb <- table(nbpred, data_test$Y)
TP <- cm_nb["Yes","Yes"]; FP <- cm_nb["Yes","No"]
FN <- cm_nb["No","Yes"];  TN <- cm_nb["No","No"]
sens <- TP / (TP + FN)
spec <- TN / (TN + FP)
sens; spec
#sensitivity : 57.5% 
#specificity : 80.4% 

#ROC-AUC
pred = prediction(nbpred2[,2], data_test$Y) #using predicted probabilities for Default = Yes
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc") # Calculate AUC // AUC = 0.77
plot(perf, col = "steelblue", lwd = 2, main="ROC for NB", cex.main = 1.8,
     cex.lab = 1.5) # Plot ROC curve
abline(0, 1, lwd = 1, lty = 2) # Add dashed diagonal line
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2)), cex = 2) # Compute AUC and add text to ROC plot.
#########################################
#ROC = 0.76
#########################################

#4. Decision Tree 

library(tree)
library(MASS)
library(rpart) 
library(ROCR)
# install.packages("rpart.plot")
library(rpart.plot)

big.tree = rpart(Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + X21 + X22 + X23, method = "class", data = data_train, minsplit = 25, cp = 0.0001, maxdepth = 30) #don't need to log predictors because regression tree is not distance based, don't need to do selection because regression tree will pick
#minsplit made a bit bigger than default (20) as data_train size is big (prevents overfitting)
#maxdepth kept at default

#number of leaves:
length(unique(big.tree$where)) #158 leaves

#cross-validation plot:
plotcp(big.tree)

#best cp:
bestcp = big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]

#pruning the tree based on best cp
best.tree = prune(big.tree,cp = bestcp)

#number of leaves after pruning:
length(unique(best.tree$where))
#11 leaves

rpart.plot(best.tree, type = 3, extra = 101, fallen.leaves = TRUE, cex = 0.5)

#use best.tree to predict on data_test
treepred = predict(best.tree, newdata = data_test, type = "prob") #predict the test observations

###### JUDGING MODEL PERFORMANCE ######
pred = prediction(treepred[,2], data_test$Y)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc_perf = performance(pred, measure = "auc")
plot(perf, col = "steelblue", lwd = 2, main="ROC for Decision Tree", cex.main = 1.8,
     cex.lab = 1.5)
abline(0, 1, lwd = 1, lty = 2)
text(0.4, 0.8, paste("AUC =", round(auc_perf@y.values[[1]], 2)), cex = 2)

#AUC for tree is 0.74