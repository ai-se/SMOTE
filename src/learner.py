from __future__ import print_function, division

from collections import Counter
import pdb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from newabcd import *


# __author__ = 'WeiFu'

class Learners(object):
  def __init__(self, clf, train_X, train_Y, predict_X, predict_Y, goal):
    self.train_X = train_X
    self.train_Y = train_Y
    self.predict_X = predict_X
    self.predict_Y = predict_Y
    self.goal = goal
    self.param_distribution = self.get_param()
    self.learner = clf

  def learn(self, F, **kwargs):
    self.learner.set_params(**kwargs)
    clf = self.learner.fit(self.train_X, self.train_Y)
    predictresult = []
    predict_Y = []
    for predict_X, actual in zip(self.predict_X, self.predict_Y):
      try:
        _predictresult = clf.predict(predict_X.reshape(1, -1))
        predictresult.append(_predictresult[0])
        predict_Y.append(actual)  # for some internal issue, we have to handle skip some bad data.
      except:
        print("one pass")
        continue
    scores = self._Abcd(predictresult, predict_Y, F)
    return scores

  def _Abcd(self, predicted, actual, F):
    """
    get performance scores. not test  yet!!! 1120
    """

    def calculate(scores):
      for i, v in enumerate(scores):
        F[uni_actual[i]] = F.get(uni_actual[i], []) + [v]
      freq_actual = [count_actual[one] / len(actual) for one in uni_actual]
      F["mean"] = F.get("mean", []) + [np.mean(scores)]
      F["mean_weighted"] = (F.get("mean_weighted", []) + [
        np.sum(np.array(scores) * np.array(freq_actual))])
      ### it seems something wrong here, mean is the same as mean_weighted .......Jan, 17
      return F

    def micro_cal(goal="Micro_F"):
      TP, FN, FP = 0, 0, 0
      for each in confusion_matrix_all_class:
        TP += each.TP
        FN += each.FN
        FP += each.FP
      PREC = TP / (TP + FP)
      PD = TP / (TP + FN)
      F[goal] = F.get(goal, []) + [2.0 * PREC / (PD + PREC)]
      return F

    def macro_cal(goal="Macro_F"):
      PREC_sum, PD_sum = 0, 0
      for each in confusion_matrix_all_class:
        PD_sum += each.stats()[0]
        PREC_sum += each.stats()[2]
      PD_avg = PD_sum/len(uni_actual)
      PREC_avg = PREC_sum/len(uni_actual)
      F[goal] = F.get(goal, []) + [2.0 * PREC_avg * PD_avg / (PREC_avg + PD_avg)]
      return F

    # pdb.set_trace()
    _goal = {"PD": 0, "PF": 1, "PREC": 2, "ACC": 3, "F": 4, "G": 5}
    abcd = ABCD(actual, predicted)
    uni_actual = list(set(actual))
    count_actual = Counter(actual)
    if "Micro" in self.goal or "Macro" in self.goal:
      confusion_matrix_all_class = [each for each in abcd()]
      if "Micro" in self.goal:
        return micro_cal()
      else:
        return macro_cal()
    else:
      score_each_klass = [k.stats()[_goal[self.goal]] for k in abcd()]  # -2 is F measure
      return calculate(score_each_klass)

  def get_param(self):
    raise NotImplementedError("You should implement get_param function")


class CartClassifier(Learners):
  name = "CART"

  def __init__(self, train_x, train_y, predict_x, predict_y, goal):
    clf = DecisionTreeClassifier()
    self.name = "CART"
    super(CartClassifier, self).__init__(clf, train_x, train_y, predict_x, predict_y, goal)

  def get_param(self):
    tunelst = {
      "max_features": [0.01, 1],
      "max_depth": [1, 50],
      "min_samples_split": [2, 20],
      "min_samples_leaf": [1, 20],
      "random_state": [1, 1]
    }
    return tunelst


class RfClassifier(Learners):
  name = "RF"

  def __init__(self, train_x, train_y, predict_x, predict_y, goal):
    clf = RandomForestClassifier()
    self.name = "RF"
    super(RfClassifier, self).__init__(clf, train_x, train_y, predict_x, predict_y, goal)

  def get_param(self):
    tunelst = {
      "min_samples_split": [1, 20],
      "min_samples_leaf ": [2, 20],
      "max_leaf_nodes": [10, 50],
      "n_estimators": [50, 150],
      "max_features": [0.01, 1],
      "random_state": [1, 1]
    }
    return tunelst


class Naive_bayes(Learners):
  name = "NB"

  def __init__(self, train_x, train_y, predict_x, predict_y, goal):
    clf = MultinomialNB()
    self.name = "NB"
    super(Naive_bayes, self).__init__(clf, train_x, train_y, predict_x, predict_y, goal)

  def get_param(self):
    tunelst = {
      "alpha": [0.0, 1.0],
      "fit_prior": [False, True]
    }
    return tunelst


class Linear_SVM(Learners):
  name = "Linear_SVM"

  def __init__(self, train_x, train_y, predict_x, predict_y, goal):
    clf = LinearSVC(dual=False)
    self.name = "Linear_SVM"
    super(Linear_SVM, self).__init__(clf, train_x, train_y, predict_x, predict_y,goal)

  def get_param(self):
    tunelst = {
      "C": [0.01, 5.0],
      # "dual":[False, True],
      # "multi_class":['ovr','crammer_singer'],
      "penalty": ['l1', 'l2'],
      # "loss":['hinge','squared_hinge'],
      "random_state": [1, 1]
    }
    return tunelst

# def linear_SVM(train_x, train_y, predict_x, predict_y, F, **kwargs):
#   clf = LinearSVC(dual=False, **kwargs)
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)
#
#
# def naive_bayes(self.train_x, train_y, predict_x, predict_y, F, **kwargs):
#   super(naive_bayes, self).__init__(train_x, train_y, predict_x,
#                                          predict_y, F)
#   clf = MultinomialNB(**kwargs)
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)

#     i.tune_min = [1, 2, 10, 50, 0.01, 0.01]
#     i.tune_max = [20, 20, 50, 150, 1, 1]

# def _Abcd(predicted, actual, F):
#   """
#   get performance scores. not test  yet!!! 1120
#   """
#
#   def calculate(scores):
#     for i, v in enumerate(scores):
#       F[uni_actual[i]] = F.get(uni_actual[i], []) + [v]
#     freq_actual = [count_actual[one] / len(actual) for one in uni_actual]
#     F["mean"] = F.get("mean", []) + [np.mean(scores)]
#     F["mean_weighted"] = (F.get("mean_weighted", []) + [
#       np.sum(np.array(scores) * np.array(freq_actual))])
#     return F
#
#   # pdb.set_trace()
#   abcd = ABCD(actual, predicted)
#   uni_actual = list(set(actual))
#   count_actual = Counter(actual)
#   score_each_klass = [k.stats()[-2] for k in abcd()]  # -2 is F measure
#   return calculate(score_each_klass)
#
#
# def learn(clf, train_X, train_Y, predict_X, predict_Y, F):
#   clf = clf.fit(train_X, train_Y)
#   predictresult = clf.predict(predict_X)
#   scores = _Abcd(predictresult, predict_Y, F)
#   return scores


# def cartClassifier(train_x, train_y, predict_x, predict_y, F, **kwargs):
#   clf = DecisionTreeClassifier(**kwargs)
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)
#
#
# def rfClassifier(train, train_x, train_y, predict_x, predict_y, F, **kwargs):
#   clf = RandomForestClassifier(**kwargs)
#   # pdb.set_trace()
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)


# def naive_bayes(train_x, train_y, predict_x, predict_y, F, **kwargs):
#   clf = MultinomialNB(**kwargs)
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)
#
#
# def linear_SVM(train_x, train_y, predict_x, predict_y, F, **kwargs):
#   clf = LinearSVC(dual=False, **kwargs)
#   return learn(clf, train_x, train_y, predict_x, predict_y, F)
#
#
# def logistic():
#   clf = linear_model.LogisticRegression()
#   return learn(clf)
