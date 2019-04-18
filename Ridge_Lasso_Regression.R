# Regularization - Ridge and Lasso regression

# Installed the required packages
gc()
rm(list = ls(all = TRUE))

packages<-function(x){
  x<-as.character(match.call()[[2]])
  if (!require(x,character.only=TRUE)){
    install.packages(pkgs=x,repos="http://cran.r-project.org")
    require(x,character.only=TRUE)
  }
}

packages(caret)
packages(caTools)
packages(psych)
packages(mlbench)
packages(glmnet)

# Here we are using Boston Housing data from package 'mlbench'

data("BostonHousing")

df = BostonHousing

str(df)

# Lets look at the correlation between independ variables
pairs.panels(df[,c(-4,-14)],
             cex = 2)

# From the regression plot, we have observed that there are independent variables which are correlated, this leads to the multicollinearity issue
# COllinearity leads to overfitting
# We will be using Ridge, Lasso and Elastic net regression technique to resolve these issues.
# 1. Ridge Regression : Shrink coefficients to non zero  values to prevent overfit, but keeps all the variables.
# 2. Lasso Regression : Shrink regression coefficients, with some of the coefficients shrunk to zero. Thus it also helps with feature selection.
# 3. ELastic Net regression : Mix of Lasso and Ridge

# SSE_Ridge = RSS + {Lambda* sum(Beta^2)} ---- Penalty term including square of coefficients
# SSE_LASSO = RSS + {Lambda* sum|Beta|} ---- Penalty term including absolute values
# SSE_Elastic_Net = RSS + Lambda {[(1-alpha)*sum(Beta^2)] + [aplha * sum|Beta|]}
# aplha can be 0 to 1


# The objective is to minimize the cost (SSE)


# Data Partitioning
set.seed(123)
id = sample.split(Y = df$medv, SplitRatio = 0.7)
train_df = subset(df, id == "TRUE")
test_df = subset(df, id == "FALSE")

# Custom control parameters - using caret package for this

custom = trainControl(method = 'repeatedcv',
                      number = 10,
                      repeats = 5,
                      verboseIter = T)
# This function used to get the cross validation

# Linear Model
set.seed(123)

lm = train(medv ~., data = train_df,
           method = "lm",
           trControl = custom )
attributes(lm)

lm$results
lm$finalModel
# If we simply type lm, we will get the model detail
lm
summary(lm)

plot(lm$finalModel)


# Lets do the Ridge regression
set.seed(123)
ridge = train(medv~., data = train_df,
              method = 'glmnet',
              tuneGrid = expand.grid(alpha = 0,lambda = seq(0.0001, 1, length = 5)),
              trControl = custom )
# This model finds the best value of Lambda (0.5), the Lambda is here is a hyper parameter.
# Lambda is estimated using cross validations. It is basically the strength of the penalty on the coefficients.
# As we increase lambda, we are increasing the penalty.decreasing the lambda, we are reducing the penalty.

# Plot the result
plot(ridge)

# Get the models detail
ridge

plot(ridge$finalModel, xvar = 'lambda', label = T)

# WHen log lambda is around 9 or 10, coefficients around zero.

plot(varImp(ridge, scale = T))


# Lets do the LASSO regression ( We need to put alpha =1 in the previous code only)
set.seed(123)
lasso = train(medv~., data = train_df,
              method = 'glmnet',
              tuneGrid = expand.grid(alpha = 1,lambda = seq(0.0001, 1, length = 5)),
              trControl = custom )

plot(lasso)
plot(lasso$finalModel, xvar = 'lambda', label = T)
plot(varImp(lasso, scale = T))


# Lets do the ELASTIC NET regression ( We need to put value of alpha in range to get the results)
set.seed(1234)
en = train(medv~., data = train_df,
              method = 'glmnet',
              tuneGrid = expand.grid(alpha = seq(0,1,length = 10),
                                     lambda = seq(0.0001, 1, length = 5)),
           trControl = custom )

plot(en)

# Lets try to change the lambda and rerun the model

set.seed(1234)
en = train(medv~., data = train_df,
           method = 'glmnet',
           tuneGrid = expand.grid(alpha = seq(0,1,length = 10),
                                  lambda = seq(0.0001, 0.2, length = 5)),
           trControl = custom )

plot(en)
