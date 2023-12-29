import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(df['landmass'].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
df_36 = df['landmass'].isin([3, 6])

#Print the average vales of the predictors for Europe and Oceania
print(df.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = (df["landmass"].isin([3,6]))*1

#Print the variable types for the predictors
print(labels)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df[var])

#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.3, random_state=8)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

for i in depths:
  dtc = DecisionTreeClassifier(max_depth=i)
  dtc.fit(x_train, y_train)
  y_pred = dtc.predict(x_test)
  score = dtc.score(x_test, y_test)
  acc_depth.append(score)
  # print(i, acc_depth)


#Plot the accuracy vs depth
plt.plot(depths, acc_depth, color='red')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title('Accuracy by max_depth')
plt.show()
plt.clf()

#Find the largest accuracy and the depth this occurs
#print(np.max(acc_depth))

#Refit decision tree model with the highest accuracy and plot the decision tree
plt.figure(figsize=(12, 12))
tree.plot_tree(dtc, feature_names = x_train.columns, filled=True, label='all')
plt.tight_layout()
plt.show()
plt.clf()
feature_names = x_train.columns.tolist()
print(tree.export_text(dtc, feature_names=feature_names))
#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
ccp_test = np.arange(0, .05, .001)
acc_pruned = []
ccp_alphas = []

for c in ccp_test:
    dt_tuned = DecisionTreeClassifier(max_depth = 3, ccp_alpha=c)
    dt_tuned.fit(x_train, y_train)
    y_pred = dt_tuned.predict(x_test)
    # print the accuracy for each ccp_alpha value
    score = dt_tuned.score(x_test, y_test)
    print(f'ccp_alpha = {c}: Accuracy = {score}')
    acc_pruned.append([score])
    ccp_alphas.append([c])

#Plot the accuracy vs ccp_alpha
plt.plot(acc_pruned, ccp_alphas, marker = 'o')
plt.xlabel('accuracy')
plt.ylabel('ccp_alpha')
plt.show()
plt.clf()
#Find the largest accuracy and the ccp value this occurs
print(np.max(acc_pruned))
## 0.6610169491525424
#Fit a decision tree model with the values for max_depth and ccp_alpha found above
dtc_tuned = DecisionTreeClassifier(max_depth = 12, ccp_alpha=0.044)
dtc_tuned.fit(x_train, y_train)
y_pred = dt_tuned.predict(x_test)
tuned_score = dtc_tuned.score(x_test, y_test)
print(f'The accuracy of the model is: {tuned_score}')

#Plot the final decision tree
plt.figure(figsize=(12, 12))
tree.plot_tree(dtc_tuned, feature_names = x_train.columns, filled=True, label='all')
plt.tight_layout()
plt.show()
plt.clf()
feature_names = x_train.columns.tolist()
print(tree.export_text(dtc, feature_names=feature_names))