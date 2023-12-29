
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

df = pd.read_csv('tennis_stats.csv')

df.info()
print(df.head())


# perform exploratory analysis here:

plt.scatter(df['Wins'], df['Winnings'], alpha=0.4)
plt.title('Wins to Winnings')
plt.xlabel('Wins')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(df['Losses'], df['Winnings'], alpha=0.4)
plt.title('Losses to Winnings')
plt.xlabel('Losses')
plt.ylabel('Winnings')
plt.show()
plt.clf()

plt.scatter(df['Wins'], df['Ranking'], alpha=0.4)
plt.title('Wins to Ranking')
plt.xlabel('Wins')
plt.ylabel('Ranking')
plt.show()
plt.clf()

plt.scatter(df['Aces'], df['Wins'], alpha=0.4)
plt.title('Aces to Wins')
plt.xlabel('Aces')
plt.ylabel('Wins')
plt.show()
plt.clf()

plt.scatter(df['DoubleFaults'], df['Losses'], alpha=0.4)
plt.title('Double Faults to Losses')
plt.xlabel('Double Faults')
plt.ylabel('Losses')
plt.show()
plt.clf()


## perform single feature linear regressions here:
## We are going to look at double faults to losses

features = df[['DoubleFaults']]
outcome = df[['Losses']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.25, random_state=4)

model = LinearRegression()
model.fit(features_train, outcome_train)
model.score(features_test, outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Double Faults to Losses - Predicted')
plt.show()
plt.clf()

features = df[['Aces']]
outcome = df[['Wins']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.25, random_state=4)

model = LinearRegression()
model.fit(features_train, outcome_train)
model.score(features_test, outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Aces to Wins - Predicted')
plt.show()
plt.clf()


## perform two feature linear regressions here:
features = df[['FirstServePointsWon',
'SecondServePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.25, random_state=4)

model = LinearRegression()
model.fit(features_train, outcome_train)
model.score(features_test, outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Points Won from Serve to Winnings - Predicted')
plt.show()
plt.clf()


features = df[['BreakPointsOpportunities',
'FirstServeReturnPointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.25, random_state=4)

model = LinearRegression()
model.fit(features_train, outcome_train)
model.score(features_test, outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Break Point Opportunities + First Serve Returns Won to Winnings - Predicted')
plt.show()
plt.clf()


## perform multiple feature linear regressions here:

features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, test_size=0.25, random_state=4)

model = LinearRegression()
model.fit(features_train, outcome_train)
model.score(features_test, outcome_test)

prediction = model.predict(features_test)
plt.scatter(outcome_test,prediction, alpha=0.4)
plt.title('Features to Winnings')
plt.xlabel('All features')
plt.ylabel('Winnings')
plt.show()
plt.clf()



