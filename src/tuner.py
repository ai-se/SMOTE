from __future__ import print_function, division
import sys
import os
import pdb
import random
import pandas as pd


# __author__ = 'WeiFu'





class BaseSearch(object):
  def __init__(self, learner, params_distribution, train_data, tune_data,
               objective):
    self.learner = learner
    self.tune_data = tune_data
    self.params_distribution = params_distribution
    self.tune_goal = objective

  def evaluate(self):
    raise NotImplementedError("Please implement evaluate")


class DE(object):
  """
  :parameter
  ===========
  :param learner: data minier to be used to predict
  :param paras_distribution: dictionary type, key is the name, value is a
  list showing range
  :param train_data: training data sets, panda.DataFrame type
  :param tune_data: tuning data sets, panda.DataFrame type
  :param goal: tuning goal, can be "PD, PF, F, PREC, G" ect
  :param num_population: num of population in DE
  :param repeats: num of repeats,
  :param life: early termination.
  :param f: prob of mutation a+f*(b-c)
  :param cr: prob of crossover
  """

  def __init__(self, params_distribution, goal="F", num_population=60,
               repeats=60, f=0.75, cr=0.3, life=5):
    self.np = num_population
    self.repeats = repeats
    self.f = f
    self.cr = cr
    self.life = life
    self.params_distribution = params_distribution
    self.tune_goal = goal
    self.evaluation = 0
    self.scores = {}
    self.frontier = [self.generate() for _ in xrange(self.np)]
    self.evaluate()
    self.bestconf, self.bestscore = self.best()

  def generate(self):
    candidate = {}
    for key, val in self.params_distribution.iteritems():
      if isinstance(val[0], float):
        candidate[key] = round(random.uniform(val[0], val[1]), 3)
      elif isinstance(val[0], bool):
        candidate[key] = random.random() <= 0.5
      elif isinstance(val[0], str):
        candidate[key] = random.choice(val)
      elif isinstance(val[0], int):
        candidate[key] = int(random.uniform(val[0], val[1]))
      elif isinstance(val[0], list) and isinstance(val[0][0], int):
        candidate[key] = [int(random.uniform(each[0], each[1])) for each in
                          val]
      else:
        raise ValueError("type of params distribution is wrong!")
    if "random_state" in self.params_distribution.keys():
      candidate["random_state"] = 1  ## set random seed here
    return candidate

  def best(self):
    raise NotImplementedError("Please implement evaluate")

  def evaluate(self):
    raise NotImplementedError("Please implement evaluate")

  def evaluate_once(self, **kwargs):
    raise NotImplementedError("Please implement evaluate")

  def get_target_score(self, score_dict):
    raise NotImplementedError()

  def gen3(self, n, f):
    seen = [n]

    def gen1(seen):
      while 1:
        k = random.randint(0, self.np - 1)
        if k not in seen:
          seen += [k]
          break
      return self.frontier[k]

    a = gen1(seen)
    b = gen1(seen)
    c = gen1(seen)
    return a, b, c

  def trim(self, n, x):
    if isinstance(self.params_distribution[n][0], float):
      return max(self.params_distribution[n][0],
                 min(round(x, 2), self.params_distribution[n][1]))
    elif isinstance(self.params_distribution[n][0], int):
      return max(self.params_distribution[n][0],
                 min(int(x), self.params_distribution[n][1]))
    else:
      raise ValueError("wrong type here in parameters")

  def update(self, index, old):
    newf = {}
    a, b, c = self.gen3(index, old)
    for key, val in old.iteritems():
      if isinstance(self.params_distribution[key][0], bool):
        newf[key] = old[key] if self.cr < random.random() else not old[key]
      elif isinstance(self.params_distribution[key][0], str):
        newf[key] = random.choice(self.params_distribution[key])
      elif isinstance(self.params_distribution[key][0], list):
        temp_lst = []
        for i, each in enumerate(self.params_distribution[key]):
          temp_lst.append(old[key][i] if self.cr < random.random() else
                          max(self.params_distribution[key][i][0],
                              min(self.params_distribution[key][i][1],
                                  int(a[key][i] + self.f * (
                                  b[key][i] - c[key][i])))))
        newf[key] = temp_lst
      else:
        newf[key] = old[key] if self.cr < random.random() else self.trim(key, (
          a[key] + self.f * (b[key] - c[key])))
    return newf

  def Tune(self):
    def isBetter(new, old):
      return new < old if self.tune_goal == "PF" else new > old

    changed = False
    for k in xrange(self.repeats):
      if self.life <= 0:
        break
      nextgeneration = []
      for index, f in enumerate(self.frontier):
        new = self.update(index, f)

        # self.assign(self.tobetuned, new)
        # newscore = self.callModel()
        newscore = self.get_target_score(self.evaluate_once(**new))
        self.evaluation += 1
        if isBetter(newscore[self.target_class],
                    self.scores[index][self.target_class]):
          nextgeneration.append(new)
          self.scores[index] = newscore
        else:
          nextgeneration.append(f)
      self.frontier = nextgeneration[:]
      newbestconf, newbestscore = self.best()
      if isBetter(newbestscore, self.bestscore):
        # print("newbestscore %s:" % str(newbestscore))
        # print("bestconf %s :" % str(newbestconf))
        self.bestscore = newbestscore
        self.bestconf = newbestconf
        changed = True
      if not changed:
        self.life -= 1
      changed = False
    # print("final bestescore %s: " + str(self.bestscore))
    # print("final bestconf %s: " + str(self.bestconf))
    # print("DONE !!!!")
    return (self.bestconf, self.evaluation)


class DE_Tune_ML(DE):
  def __init__(self, learner, params_distribution, target_class, goal,
               num_population=60, repeats=60, f=0.75, cr=0.3, life=5):
    self.learner = learner
    self.target_class = target_class
    super(DE_Tune_ML, self).__init__(params_distribution, goal, num_population,
                                     repeats, f, cr, life)

  def evaluate(self):
    # clf = self.learner(self.train_X, self.train_Y, self.test_X, self.test_Y,
    #                    {}, self.tune_goal)
    for n, kwargs in enumerate(self.frontier):
      score_dict = self.learner.learn({}, **kwargs)
      self.scores[n] = self.get_target_score(score_dict)
      # each return value like this{"mean":0.2,"weighted_mean":0.9}

  def evaluate_once(self,**new):
    return self.learner.learn({}, **new)

  def get_target_score(self, score_dict):
    temp = {}
    # pdb.set_trace()
    for key, val in score_dict.iteritems():
      if key == self.target_class:
        temp[key] = val[0]  # value, not list
    return temp

  def best(self):
    sortlst = []
    # pdb.set_trace()
    if self.tune_goal == "PF":  # the less, the better.
      sortlst = sorted(self.scores.items(),
                       key=lambda x: x[1][self.target_class], reverse=True)
    else:
      sortlst = sorted(self.scores.items(),
                       key=lambda x: x[1][self.target_class])
    bestconf = self.frontier[sortlst[-1][0]]
    bestscore = sortlst[-1][-1]
    return bestconf, bestscore


class DE_Tune_SMOTE(DE):
  def __init__(self, learner, smote, params_distribution, train_pd, tune_pd,
               target_class, goal, num_population=10, repeats=60,
               f=0.75, cr=0.3, life=5):
    self.learner = learner  ## pass the class, not fitted yet
    self.train = train_pd
    self.tune = tune_pd
    self.smote = smote
    self.target_class = target_class
    super(DE_Tune_SMOTE, self).__init__(params_distribution, goal,
                                        num_population, repeats, f, cr, life)

  def evaluate(self):
    for n, kwargs in enumerate(self.frontier):
      k = kwargs["k"]
      up_to_num = kwargs["up_to_num"]
      score_lst = self.fit_learner(k, up_to_num)
      self.scores[n] = self.get_target_score(score_lst)

  def evaluate_once(self, **kwargs):
    k = kwargs["k"]
    up_to_num = kwargs["up_to_num"]
    return self.fit_learner(k, up_to_num)

  def fit_learner(self, k, up_to_num):
    train_smoted = self.smote(self.train, k, up_to_num).run()
    train_X, train_Y = self.get_XY(train_smoted)
    tune_X, tune_Y = self.get_XY(self.tune)
    F = self.learner(train_X, train_Y, tune_X, tune_Y).learn({})
    return F

  def get_XY(self, data):
    X = data.ix[:, data.columns[:-1]].values
    Y = data.ix[:, data.columns[-1]].values
    return X, Y

  def get_target_score(self, score_dict):
    temp = {}
    for key, val in score_dict.iteritems():
      if key in self.target_class:
        temp[key] = val[0]  # value, not list
    return temp

  def best(self):
    sortlst = []
    if self.tune_goal == "PF":  # the less, the better.
      sortlst = sorted(self.scores.items(),
                       key=lambda x: x[1][self.target_class], reverse=True)
    else:
      sortlst = sorted(self.scores.items(),
                       key=lambda x: x[1][self.target_class])
    bestconf = self.frontier[sortlst[-1][0]]
    bestscore = sortlst[-1][-1]
    return bestconf, bestscore
