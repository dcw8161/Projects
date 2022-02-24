library(readr)
Red_Cross_Donation_data <- read_csv("Desktop/DDSA/Projects/Red Cross/Red Cross Donation data.csv")
View(Red_Cross_Donation_data)

#Making a local copy of dataset
df = Red_Cross_Donation_data

#Viewing the contents of the data file
View(df)

nrow(df) # to get number of rows
ncol(df) # to get number of columns
dim(df) # display dimensions of the Dataset

colnames(df) # display columns names

# Summary of the dataset
summary(df)

# Data Structure of the data set
str(df)

# Option 1 - Doing detailed Exploratory Data Analysis using Datamaid package
#install.packages("dataMaid") # run only once per session
library(dataMaid)
makeDataReport(df, file = "RC_Exploratory_Report.html")

# Option 2  - Doing detailed Exploratory Data Analysis using DataExplorer package
#install.packages("DataExplorer") # run only once per session
library(DataExplorer)
create_report(df)

#To take out noise variables 
#deployer select everything (df) except noise variables with minus sign 
library(dplyr)
df1 = select(df, -'MARITAL_STATUS',-'WEALTH_RATING', -'DEGREE_LEVEL', -'BIRTH_DATE', -'ID', -'MEMBERSHIP_IND', -'ALUMNUS_IND')
View(df1)

# Handling missing data
# Loading the AI package for machine learning - healthcareAI
install.packages("healthcareai")
library(healthcareai)

#Checking the Missing data proportion
missingness(df1)

# Replacing numerical values with Median
#df1$""[is.na(df1[[""]])] = median(df1$"", na.rm = TRUE)
#df1$""[is.na(df1[[""]])] = median(df1$"", na.rm = TRUE)

# Setting Categorical variables
df1$"ZIPCODE" = as.factor(df1$"ZIPCODE")
df1$"GENDER" = as.factor(df1$"GENDER")
df1$"PREF_ADDRESS_TYPE" = as.factor(df1$"PREF_ADDRESS_TYPE")
library(readr)
df1$'PrevFYGiving' = parse_number(df1$'PrevFYGiving') 
df1$'PrevFY1Giving'= parse_number(df1$'PrevFY1Giving')
df1$'PrevFY2Giving'= parse_number(df1$'PrevFY2Giving')
df1$'PrevFY3Giving'= parse_number(df1$'PrevFY3Giving')
df1$'PrevFY4Giving'= parse_number(df1$'PrevFY4Giving')
df1$'CurrFYGiving'= parse_number(df1$'CurrFYGiving')
View(df1)
# combining fiscalyear givings to make 'CONSECUTIVEYRG' column
df1$'CONSECUTIVEYRG'=(df1$'PrevFYGiving'+ df1$'PrevFY1Giving' + df1$'PrevFY2Giving' + df1$'PrevFY3Giving'
   + df1$'PrevFY4Giving'+df1$'CurrFYGiving')
View(df1)

# Taking out redundant non-categorical variables
df2 = select(df1, -"GENDER", -'PrevFYGiving', -'PrevFY1Giving', 
             -'PrevFY2Giving',  -'PrevFY3Giving', -'PrevFY4Giving', 
             -'CurrFYGiving', -'TotalGiving')
View(df2)
# Training the machine with algorithms
#ml_rc_classif = machine_learn(df2, outcome = 'DONOR_IND')
# Training the machine with Generalized Linear model algorithms
glm_models = machine_learn(df2,outcome= 'DONOR_IND', models= "glm")

# Evaluating the Accuracy
evaluate(glm_models)

# Training the machine with Ensemble Tree algorithms
rf_models = machine_learn(df2, outcome='DONOR_IND', models= "rf")
# Evaluating the Accuracy
evaluate(rf_models)

# Training the machine with the computationally intensive XG Boost algorithms
xgb_models = machine_learn(df2,outcome='DONOR_IND', models= "xgb")
# Evaluating the Accuracy
evaluate(xgb_models)

# Comparing the all the models
evaluate(glm_models)
evaluate(rf_models)
evaluate(xgb_models)

# Training with GLM, XGB and Ensemble algorithms in one shot (warning - computationally intensive)
#models = machine_learn(df2,outcome='DONOR_IND')
# Getting the result of all models attempted
#evaluate(models, all_models = TRUE)

# Getting the variable importance for predictions
get_variable_importance(xgb_models) # Based on the Extreme Gradient Algo
get_variable_importance(rf_models) # Based on the Random Forest algo
#pass test data to use this model - import and run 
#predict()
#dim(df2)
library(healthcareai)
df2$DONOR_IND <- factor(df2$DONOR_IND) 
d <- split_train_test(df2, "DONOR_IND", .75)
df_ml=machine_learn(d$train, outcome="DONOR_IND")
predictions <- predict(df_ml, newdata = d$test)
predictions
evaluate(predictions)
plot(predictions)
class_preds <- predict(df_ml, newdata = d$test, outcome_groups = 10)
table(actual = class_preds$DONOR_IND, predicted = class_preds$predicted_group)