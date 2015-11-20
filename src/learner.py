from __future__ import print_function, division
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from settings import *
 # __author__ = 'WeiFu'


def learn(clf):
  pass

def cart():
  clf = DecisionTreeRegressor(
    max_features=The.cart.max_features,
    max_depth=The.cart.max_depth,
    min_samples_split=The.cart.min_samples_split,
    min_samples_leaf=The.cart.min_samples_leaf,
    random_state=1)
  return learn(clf)


def cartClassifier():
  clf = DecisionTreeClassifier(
    max_features=The.cart.max_features,
    max_depth=The.cart.max_depth,
    min_samples_split=The.cart.min_samples_split,
    min_samples_leaf=The.cart.min_samples_leaf,
    random_state=1)
  return learn(clf)

def rf():
  clf = RandomForestRegressor(
    n_estimators=The.rf.n_estimators,
    max_features=The.rf.max_features,
    min_samples_split=The.rf.min_samples_split,
    min_samples_leaf=The.rf.min_samples_leaf,
    max_leaf_nodes=The.rf.max_leaf_nodes,
    random_state=1)
  return learn(clf)


def rfClassifier():
  clf = RandomForestClassifier(
    n_estimators=The.rf.n_estimators,
    max_features=The.rf.max_features,
    min_samples_split=The.rf.min_samples_split,
    min_samples_leaf=The.rf.min_samples_leaf,
    max_leaf_nodes=The.rf.max_leaf_nodes,
    random_state=1)
  # pdb.set_trace()
  return learn(clf)


def bayes():
  clf = GaussianNB()
  return learn(clf)


def logistic():
  clf = linear_model.LogisticRegression()
  return learn(clf)