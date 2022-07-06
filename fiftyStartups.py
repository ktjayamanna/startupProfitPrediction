import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
plt.close('all')
#%% import the dataset
df = pd.read_csv("./50_Startups.csv")
#%% Seperate independent variables and the dependent variable.
x = df.iloc[:, : -1]
y = df.iloc[:, -1 :]
#%% Check the correlation between dependent variables and ind. variables.
plt.figure
sns.pairplot(df, x_vars= x, y_vars= y, hue = "State")
plt.savefig('./plots/correlation.png')

#%%checkout the multicollinearity 
plt.figure
sns.pairplot(x.iloc[:, :3])
plt.savefig('./plots/multicollinearity.png')


#%% check the peak to peak values of the features
'''
We have numerical features for the most part except "state" which is nominal.
So, we have to one hot encode this. Remember that we only need two digits to 
represent ['New York', 'California', 'Florida'].
'''
state = pd.get_dummies(x['State'], drop_first = True)
#Now we drop the existing state variable and add the newly created back
x = x.drop('State', axis = 1)
x = pd.concat([x, state], axis = 1)
#%% check the peak to peak values of each feature
cols = x.columns
def get_range(seriesIn):
    maxVal = seriesIn.max()
    minVal = seriesIn.min()
    return maxVal - minVal
peak2peak = [get_range(x[i]) for i in cols]
#%%
def normalize_feature(featureLabelIn):
    maxVal = x[featureLabelIn].max()
    x[featureLabelIn] = x[featureLabelIn] / maxVal
for i in range(3):
    normalize_feature(cols[i])
#%% Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, 
                                                    random_state=5)
#%%
model = LinearRegression()
model.fit(x_train, y_train)
optimalW = model.coef_
optimalB = model.intercept_
#%% Make predictions and calcuate the r2 score
yPred= model.predict(x_test)
r2Score = r2_score(y_test, yPred)
print(f"R^2 score is {round(r2Score * 100, 2)}%")
#%%
def plot_features(FeatureIdxIn):
    selectedFeature = cols[FeatureIdxIn]
    plt.figure()
    # Plot the linear fit
    plt.plot(x_train[selectedFeature],(optimalW[0][FeatureIdxIn] * x_train[selectedFeature] + optimalB ),
             c = "g",label = f'Fitted line for {selectedFeature}')
    plt.scatter(x_test[selectedFeature], yPred, c = "b", label = "Model Predictions")
    # Create a scatter plot of the data. 
    plt.scatter(x_test[selectedFeature], y_test, marker='x', c='r', label = "Target Labels") 
    # Set the title
    plt.title(f"Startup Return of Investment with Respect to {selectedFeature}")
    # Set the y-axis label
    plt.ylabel('profits in USD')
    # Set the x-axis label
    plt.xlabel(selectedFeature)
    #Set the legend
    plt.legend()
    plt.savefig(f'./plots/{selectedFeature}.png')

for featureIdx in range(len(cols[0:3])):
    plot_features(featureIdx)




