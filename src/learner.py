from __future__ import print_function, division
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from settings import *
 # __author__ = 'WeiFu'



class Learner(object):
  def __init__(i, train, tune, test):
    i.train = train
    i.tune = tune
    i.test = test

  def untuned(i):
    The.data.predict = i.test
    The.data.train = i.train
    i.default()
    The.option.tuning = False
    score = i.call()
    return score

  def tuned(i):
    The.data.predict = i.tune
    The.data.train = i.train
    The.option.tuning = True
    i.optimizer()
    The.option.tuning = False
    The.data.predict = i.test
    score = i.call()
    return score

  def call(i):
    raise NotImplementedError

  def optimizer(i):
    raise NotImplementedError

  def default(i):
    raise NotImplementedError


class CART(Learner):
  def __init__(i, train, tune, predict):
    super(CART, i).__init__(train, tune, predict)
    i.tunelst = ["The.cart.max_features",
                 "The.cart.max_depth",
                 "The.cart.min_samples_split",
                 "The.cart.min_samples_leaf",
                 "The.option.threshold"]
    i.tune_min = [0.01, 1, 2, 1, 0.01]
    i.tune_max = [1, 50, 20, 20, 1]

  def default(i):
    The.cart.max_features = None
    The.cart.max_depth = None
    The.cart.min_samples_split = 2
    The.cart.min_samples_leaf = 1
    The.option.threshold = 0.5

  def call(i):
    return cart()

  def optimizer(i):
    tuner = CartDE(i)
    tuner.DE()


class CART_clf(CART):
  def __init__(i, train, tune, predict):
    super(CART_clf, i).__init__(train, tune, predict)

  def call(i):
    return cartClassifier()


class RF(Learner):
  def __init__(i, train, tune, predict):
    super(RF, i).__init__(train, tune, predict)
    i.tunelst = ["The.rf.min_samples_split",
                 "The.rf.min_samples_leaf ",
                 "The.rf.max_leaf_nodes",
                 "The.rf.n_estimators",
                 "The.rf.max_features",
                 "The.option.threshold"]
    i.tune_min = [1, 2, 10, 50, 0.01, 0.01]
    i.tune_max = [20, 20, 50, 150, 1, 1]
    i.default_value = [2,1,None, 100,"auto",0.5]

  def default(i):
    # for key,val in zip(i.tunelst,i.default_value):
    #   setattr(key[],key[4:],val)
    # pdb.set_trace()
    The.option.threshold = 0.5
    The.rf.max_features = "auto"
    The.rf.min_samples_split = 2
    The.rf.min_samples_leaf = 1
    The.rf.max_leaf_nodes = None
    The.rf.n_estimators = 100

  def call(i): return rf()

class RF_clf(RF):
  def __init__(i,train,tune,predict):
    super(RF_clf, i).__init__(train,tune,predict)

  def call(i): return rfClassifier()


  def optimizer(i):
    tuner = RfDE(i)
    tuner.DE()

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