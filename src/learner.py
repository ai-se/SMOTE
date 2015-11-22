from __future__ import print_function, division
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from settings import *
from newabcd import *
from collections import Counter
import numpy as np
import pdb
 # __author__ = 'WeiFu'


#
# class Learner(object):
#   def __init__(i, train, tune, test):
#     i.train = train
#     i.tune = tune
#     i.test = test
#     i.predict = None
#
#   def untuned(i):
#     i.default()
#     i.predict = i.test
#     score = i.call()
#     return score
#
#   def tuned(i):
#     i.predict = i.tune
#     i.optimizer()
#     The.option.tuning = False
#     The.data.predict = i.test
#     score = i.call()
#     return score
#
#   def call(i):
#     raise NotImplementedError
#
#   def optimizer(i):
#     raise NotImplementedError
#
#   def default(i):
#     raise NotImplementedError
#
#   def learn(i, clf, train_X, train_Y, predict):
#
#     clf = clf.fit(train_X, train_Y)
#     array = clf.predict(test_X)
#     predictresult = [i for i in array]
#     scores = _Abcd(predictresult, test_Y)
#     return scores
#
#
#
#
# class CART(Learner):
#   def __init__(i, train, tune, predict):
#     super(CART, i).__init__(train, tune, predict)
#     i.tunelst = ["The.cart.max_features",
#                  "The.cart.max_depth",
#                  "The.cart.min_samples_split",
#                  "The.cart.min_samples_leaf",
#                  "The.option.threshold"]
#     i.tune_min = [0.01, 1, 2, 1, 0.01]
#     i.tune_max = [1, 50, 20, 20, 1]
#
#   def default(i):
#     The.cart.max_features = None
#     The.cart.max_depth = None
#     The.cart.min_samples_split = 2
#     The.cart.min_samples_leaf = 1
#     The.option.threshold = 0.5
#
#   def call(i):
#     return cart(i.train, i.predict)
#
#   def optimizer(i):
#     tuner = CartDE(i)
#     tuner.DE()
#
#
# class CART_clf(CART):
#   def __init__(i, train, tune, predict):
#     super(CART_clf, i).__init__(train, tune, predict)
#
#   def call(i):
#     return cartClassifier(i.train, i.predict)
#
#
# class RF(Learner):
#   def __init__(i, train, tune, predict):
#     super(RF, i).__init__(train, tune, predict)
#     i.tunelst = ["The.rf.min_samples_split",
#                  "The.rf.min_samples_leaf ",
#                  "The.rf.max_leaf_nodes",
#                  "The.rf.n_estimators",
#                  "The.rf.max_features",
#                  "The.option.threshold"]
#     i.tune_min = [1, 2, 10, 50, 0.01, 0.01]
#     i.tune_max = [20, 20, 50, 150, 1, 1]
#     i.default_value = [2,1,None, 100,"auto",0.5]
#
#   def default(i):
#     # for key,val in zip(i.tunelst,i.default_value):
#     #   setattr(key[],key[4:],val)
#     # pdb.set_trace()
#     The.option.threshold = 0.5
#     The.rf.max_features = "auto"
#     The.rf.min_samples_split = 2
#     The.rf.min_samples_leaf = 1
#     The.rf.max_leaf_nodes = None
#     The.rf.n_estimators = 100
#
#   def call(i): return rf(i.train, i.predict)
#
# class RF_clf(RF):
#   def __init__(i,train,tune,predict):
#     super(RF_clf, i).__init__(train,tune,predict)
#
#   def call(i): return rfClassifier(i.train, i.predict)
#
#
#   def optimizer(i):
#     tuner = RfDE(i)
#     tuner.DE()


def _Abcd(predicted, actual, F):
  """
  get performance scores. not test  yet!!! 1120
  """
  def calculate(scores):
    for i, v in enumerate(scores):
      F[uni_actual[i]] = F.get(uni_actual[i],[]) +[v]
    freq_actual = [count_actual[one]/len(actual) for one in uni_actual]
    F["mean"] = F.get("mean",[]) + [np.mean(scores)]
    F["mean_weighted"] = (F.get("mean_weighted",[]) +
                         [np.sum(np.array(scores)*np.array(freq_actual))])
    return F
  # pdb.set_trace()
  abcd = ABCD(actual, predicted)
  uni_actual = list(set(actual))
  count_actual = Counter(actual)
  score_each_klass = [ k.stats()[-2]for k in abcd()]  # -2 is F measure
  return calculate(score_each_klass)


def learn(clf, train_X, train_Y, predict_X, predict_Y, F):
  clf = clf.fit(train_X, train_Y)
  predictresult = clf.predict(predict_X)
  scores = _Abcd(predictresult, predict_Y,F)
  return scores


def cartClassifier(train_x, train_y, predict_x, predict_y, F):
  clf = DecisionTreeClassifier(
    max_features=The.cart.max_features,
    max_depth=The.cart.max_depth,
    min_samples_split=The.cart.min_samples_split,
    min_samples_leaf=The.cart.min_samples_leaf,
    random_state=1)
  return learn(clf, train_x, train_y, predict_x, predict_y, F)


def rfClassifier(train, train_x, train_y, predict_x, predict_y, F):
  clf = RandomForestClassifier(
    n_estimators=The.rf.n_estimators,
    max_features=The.rf.max_features,
    min_samples_split=The.rf.min_samples_split,
    min_samples_leaf=The.rf.min_samples_leaf,
    max_leaf_nodes=The.rf.max_leaf_nodes,
    random_state=1)
  # pdb.set_trace()
  return learn(clf, train_x, train_y, predict_x, predict_y, F)


def naive_bayes(train_x, train_y, predict_x, predict_y, F):
  clf = MultinomialNB()
  return learn(clf,train_x, train_y, predict_x, predict_y, F)


def linear_SVM(train_x, train_y, predict_x, predict_y, F):
  clf = svm.LinearSVC(dual=False)
  return learn(clf,train_x, train_y, predict_x, predict_y, F)


def logistic():
  clf = linear_model.LogisticRegression()
  return learn(clf)