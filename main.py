import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np
import seaborn as sns

# Creating a function to get a Standerdized Value
def standerdized_values(val):
    return (val - val.mean())/(val.std())

df = pd.read_csv("/Users/atulat/Documents/Fish Weight Prediction/Fish.csv")

y = df["Weight"]
x = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
X = ['Length1', 'Length2', 'Length3', 'Height', 'Width']

# checking co-linearity of the dependent and the independent feature
for i in x:
    plt.figure(figsize = (7,5), facecolor = "lightblue")
    plt.scatter(df[i], y)
    plt.title(i)
    plt.show()

# since, they are co-linear, let us get into the model building

# Model 1 with y and x
model1 = smf.ols('y ~ x', data = df).fit()
# print(model1.summary())

# checking co-linearity between the independent features
for i in range(0,len(X)-1):
    for j in range(i+1,len(X)):
        plt.figure(figsize=(7, 5), facecolor="lightblue")
        plt.scatter(df[X[i]], df[X[j]])
        plt.suptitle(X[i])
        plt.title(X[j])
        plt.show()
# since, all the features are dependent to each other, we can proceed with the model, but the inference is not gonna
# make much

# The pairplot of the data
sns.pairplot(data = df)
plt.show()

# VIF Calculation
l1 = smf.ols("Length1 ~ Length2", data = df).fit().rsquared
len_vif = 1 / (1 - l1)
print("R Squared Value : ", l1)
print("VIF Value : ", len_vif)

# Quantile Quantile plot 
qqplot = sm.qqplot(model1.resid, line ='q')
plt.show()

# Plot to find the Homoscadacity
plt.scatter(standerdized_values(model1.resid), standerdized_values(model1.fittedvalues))
plt.show()


#Error Check

# fig = plt.figure(figsize=(15,8))
# fig = sm.graphics.plot_regress_exog(model1, "Height", fig=fig)
# plt.show()

