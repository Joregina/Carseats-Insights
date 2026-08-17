############################################################ 
# STAT605 Practical 2: Regression Trees                 
############################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(h2o)
library(MLmetrics)
library(recipes)
library(dplyr)
library(caTools)
library(caret)        
library(ggplot2)
library(tidyr)

#install.packages("ISLR")
library(ISLR) # This package includes a range of data sets for an introduction to statistical learning with applications in R
library(rpart) # to fit a reg tree (not in h2o)
library(rpart.plot) # to visualize the tree from rpart


# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# User-specified parameters
seed = 605     # Seed for reproducibility (students can change)
train_frac <- 0.7     # Proportion of data in training set
metric <- "rmse"      # Performance metric: residual_deviance, mae, r2, mean_residual_deviance, rmsle, rmse, null_deviance, mse
folds <- 5 # for 5-fold CV, or change to 10


# ----------------------------------------------------------------------------- #
# 2. Load & Inspect Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# For this demonstration, we will use the simulated Carseats data set. This contains sales of child car seats at 400 different stores, and includes both quantitative and qualitative variables.

data("Carseats") #Click on data object to view the dataset

?Carseats
View(Carseats)

df <- Carseats

names(df) # to get a list of the variable names in the data set

str(df) # determine what each type of variable is

# The Carseats data includes qualitative predictors (such as Shelveloc) which are already correctly specified as factor variables.   

# We wish to predict the sales of the carseats based on all of the attributes in the data set.

summary(df)

#Since a regression tree is a non-parametric method, we do not need to transform the target variable in any way.  Assumption of normality does not need to be met.


# Specify target and features

df_final <- df  # no variables need to be removed

target <- "Sales" 

# ----------------------------------------------------------------------------- #
# 3. Train/Test Split --------------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# Remember the following is generic code so nothing needs to change in this section.
set.seed(seed)  
split=sample.split(df_final[[target]],SplitRatio = train_frac) 

training_set=subset(df_final,split==TRUE) 
test_set=subset(df_final,split==FALSE)

# ----------------------------------------------------------------------------- #
# 4. Data preprocessing (outside H2O) -----------------------------------------
# ----------------------------------------------------------------------------- #

# NOTE: Regression trees (like decision trees) do not require attributes to be normalized or dummy variable encoded, however to fit a regression tree using h20, we make use of the h20.gbm function which is actually for fitting ensemble methods like random forests which DO require preprocessing of the attributes..

#We will do this purely for illustration.  It is advisable to rather fit a SINGLE regression tree using the rpart package (simpler)

# This is generic code as well, do not change

# 1. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(as.formula(paste(target, "~ .")), data = training_set) %>%
  step_normalize(all_numeric_predictors()) %>%  # this must be done first
  step_dummy(all_nominal_predictors(), one_hot = FALSE) # the last category is dropped

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE. 

# 2. Prep the recipe using the training data
rec_prep <- prep(rec, training = training_set)

# 3. Apply (bake) the prepped recipe on the scaled training set 
train_processed <- bake(rec_prep, new_data = NULL)

# 4. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_set)

summary(train_processed)
summary(test_processed)

# we do not need to scale the target for these methods

## NB: As there are categorical variables in this example, dummy variable encoding changes their names. So we need to extract the column names in the processed training/test set (train_processed) as our list of final feature names:

features <- setdiff(names(train_processed), target)


# ----------------------------------------------------------------------------- #
# 5. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# Convert scaled data to H2O dataframe
train_h2o <- as.h2o(train_processed)
test_h2o  <- as.h2o(test_processed)

# ----------------------------------------------------------------------------- #
# 6. Regression tree in H2O --------------------------------------------
# ----------------------------------------------------------------------------- #

# Hyperparameter grid for pruning/tuning
# (max_depth = tree depth, min_rows = minimum observations per leaf)
hyper_params_tree <- list(
  max_depth = seq(1, 10, 1),
  min_rows  = c(1, 5, 10)
)

#If not tuned, the default value for min_rows=10, and for max_depth=5

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian" 
)


grid_tree <- h2o.grid(
  algorithm = "gbm",
  grid_id = "reg_tree_grid", # this is just the ID we are giving the search grid
  x = features,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_tree,
  search_criteria = search_criteria,
  ntrees = 1, # ntrees=1 for single tree
  sample_rate = 1,  # no row subsampling
  col_sample_rate = 1,  # no column subsampling
  seed = seed
)

# Get the best model from the grid (based on specified metric) - note: if we want to minimize the metric then we use decreasing = FALSE, if we want to maximize the metric, then we set decreasing = TRUE

sorted_grid_tree <- h2o.getGrid("reg_tree_grid", 
                                sort_by = metric, 
                                decreasing = FALSE)

print(sorted_grid_tree)

best_model_id_tree <- sorted_grid_tree@model_ids[[1]]

best_model_tree <- h2o.getModel(best_model_id_tree)

#Extract the tuned hyperparameter(s)
tuned_max_depth <- best_model_tree@allparameters$max_depth
tuned_min_rows <- best_model_tree@allparameters$min_rows

# refit the regression tree on training set using tuned hyperparameters:

reg_tree_h2o <- h2o.gbm(
  x = features,
  y = target,
  training_frame = train_h2o,
  ntrees = 1,                
  max_depth = tuned_max_depth, 
  min_rows = tuned_min_rows,    
  sample_rate = 1,           # use all rows
  col_sample_rate = 1,       # use all predictors
  seed = seed
)

# Save predictions
preds_tree_h2o_train <- h2o.predict(reg_tree_h2o, train_h2o)
preds_tree_h2o_test <- h2o.predict(reg_tree_h2o, test_h2o)

# Convert predictions to R vector to extract from H2O environment:
preds_tree_h2o_train <- as.vector(as.data.frame(preds_tree_h2o_train)$predict)
preds_tree_h2o_test <- as.vector(as.data.frame(preds_tree_h2o_test)$predict)

# Look at performance:
MAE(preds_tree_h2o_train,train_processed[[target]])  
RMSE(preds_tree_h2o_train,train_processed[[target]])  
R2_Score(preds_tree_h2o_train,train_processed[[target]]) 

## Performance on test set:

MAE(preds_tree_h2o_test,test_processed[[target]])
RMSE(preds_tree_h2o_test,test_processed[[target]])
R2_Score(preds_tree_h2o_test,test_processed[[target]])


# ----------------------------------------------------------------------------- #
# 7. Regression tree in rpart --------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed) #always need to set the seed again before fitting any model for reproducibility of results, as rpart automatically runs cross-validation in the background. 

# Note: rpart itself doesn’t have built-in grid search

tree_rpart <- rpart(as.formula(paste(target, "~ .")),  
                    data = training_set,  # fit to the unprocessed training set
                    method  = "anova", # for regression tree
                    xval = folds, # xval argument = number of folds for CV
                    control = rpart.control(
                      #cp = 0.01,             # complexity parameter for weakest link pruning
                      minsplit = 20,         # default value for minimum observations to attempt a split
                      maxdepth = 30           # default value for maximum depth
                    )
)
?rpart
?rpart.control

# In H2O, trees are not pruned post-hoc. Instead, H2O controls tree growth while building the tree using: max_depth and min_rows. rpart however grows a large tree, then prunes back using cp and CV.

#By default, rpart function runs 10-fold cross validation (or k-folds where xval = k has been specified in the rpart function. Default stopping criterion is that a node needs to contain a minimum of 20 observations in order for further splitting to be permitted.

tree_rpart # run this to see information about the fitted tree
summary(tree_rpart) #you can run this too, but it provides TOO MUCH information!

# We can obtain the table and plot the error of the tree according to the different values of cp (cost complexity parameter):

# Value of cp is inversely related to the complexity of the tree.
# cp is tuned by examining how SSE on the training set diminishes with increasing complexity.  
# Eventually diminishing returns will set in i.e. it doesn't make sense to grow the tree any further


tree_rpart$cptable

# rel error (Relative Error): This is the training error at a given level of pruning, relative to the root node
# xerror (Cross-validated Error): This is the estimated prediction error obtained via k-fold cross-validation
# xstd (Standard Deviation of xerror): This is the standard deviation of the cross-validated error across folds.

plotcp(tree_rpart)
# y-axis is relative cross validation error, lower x-axis is cost complexity (alpha) value, upper x-axis is the number 
# of terminal nodes (tree size = T). You may also notice the dashed line. Breiman et al. (1984) 
# suggested that in actual practice, its better to instead use the smallest tree within 1 standard 
# deviation of the minimum cross validation error (aka the 1-SE rule). This is what the dashed line 
# represents. Thus, we should aim to use the value of cp furthest to the left that corresponds with a xval error below the dashed line to determine an appropriate
# smaller size (pruned) tree.

# print the tuned cp:
tree_rpart$cptable[which.min(tree_rpart$cptable[,"xerror"]), "CP"]

# ----------------------------------------------------------------------------- #
# 8. Visualizing the rpart tree --------------------------------------------
# ----------------------------------------------------------------------------- #

# We can visualize our model with rpart.plot

rpart.plot(tree_rpart)

rpart.plot(tree_rpart, yesno=1,type=2,fallen.leaves = FALSE) # add additional options to change the appearance.

#fallen.leaves = FALSE means the terminal nodes are not forced to the bottom of the plot.

# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot

# ----------------------------------------------------------------------------- #
# 9. Performance metrics for rpart tree --------------------------------------------
# ----------------------------------------------------------------------------- #

preds_tree_rpart_train = predict(tree_rpart,newdata=training_set) 
preds_tree_rpart_test = predict(tree_rpart,newdata=test_set) 

# Look at performance:
MAE(preds_tree_rpart_train,training_set[[target]])  
RMSE(preds_tree_rpart_train,training_set[[target]])  
R2_Score(preds_tree_rpart_train,training_set[[target]]) 

## Performance on test set:

MAE(preds_tree_rpart_test,test_set[[target]])
RMSE(preds_tree_rpart_test,test_set[[target]])
R2_Score(preds_tree_rpart_test,test_set[[target]])

# Pruned tree (Performance will be worse on training set)
tree_rpart2 <- rpart(as.formula(paste(target, "~ .")),  
                     data = training_set,  
                     method  = "anova", cp = 0.01277845            
)

rpart.plot(tree_rpart2, yesno=1,type=2,fallen.leaves = FALSE)



# variable importance

caret::varImp(tree_rpart) # we use the VarImp function to extract the overall variable importance.  

# ----------------------------------------------------------------------------- #
# 10. Shutdown H2O ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# If a mistake was made along the way or H2O model needed to be run again, shut H2o down and start again.

h2o.shutdown(prompt = FALSE)
