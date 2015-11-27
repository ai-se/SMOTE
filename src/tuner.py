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
      elif isinstance(val[0], list):
        pass
      elif isinstance(val[0], int):
        candidate[key] = int(random.uniform(val[0], val[1]))
      else:
        raise ValueError("type of params distribution is wrong!")
    candidate["random_state"] = 1  ## set random seed here
    return candidate

  def best(self):
    raise NotImplementedError("Please implement evaluate")

  def evaluate(self):
    raise NotImplementedError("Please implement evaluate")
  
  
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
    pdb.set_trace()
    if isinstance(self.params_distribution[n][0], float):
      return max(self.params_distribution[n][0], min(round(x, 2), self.params_distribution[n][1]))
    elif isinstance(self.params_distribution[n][0], int):
      return max(self.params_distribution[n][0], min(int(x), self.params_distribution[n][1]))
    else:
      raise ValueError("wrong type here in parameters")

  def update(self, index, old):
    newf = []
    a, b, c = self.gen3(index, old)
    for k in xrange(len(old)):
      if isinstance(self.params_distribution[k], bool):
        newf.append(old[k] if self.cr < random.random() else not old[k])
      elif isinstance(self.params_distribution[k], list):
        pass
      else:
        newf.append(old[k] if self.cr < random.random() else self.trim(k, (a[k] + self.fa * (b[k] - c[k]))))
    return newf

  def Tune(self):
    changed = False
    def isBetter(new, old):
      return new < old if self.tune_goal == "PF" else new > old

    for k in xrange(self.repeats):
      if self.life <= 0:
        break
      nextgeneration = []
      for index, f in enumerate(self.frontier):
        new = self.update(index, f)
        self.assign(self.tobetuned, new)
        newscore = self.callModel()
        self.evaluation += 1
        if isBetter(newscore[self.obj], self.scores[index][self.obj]):
          nextgeneration.append(new)
          self.scores[index] = newscore[:]
        else:
          nextgeneration.append(f)
      self.frontier = nextgeneration[:]
      newbestconf, newbestscore = self.best()
      if isBetter(newbestscore, self.bestscore):
        print
        "newbestscore %s:" % str(newbestscore)
        print
        "bestconf %s :" % str(newbestconf)
        self.bestscore = newbestscore
        self.bestconf = newbestconf[:]
        changed = True
      if not changed:
        self.life -= 1
      changed = False
    self.assign(self.tobetuned, self.bestconf)
    self.writeResults()
    print
    "final bestescore %s: " + str(self.bestscore)
    print
    "final bestconf %s: " + str(self.bestconf)
    print
    "DONE !!!!"


class DE_Tune_ML(DE):
  def __init__(self, learner, params_distribution, target_class="", goal="F",
               num_population=60, repeats=60, f=0.75, cr=0.3, life=5):
    self.learner = learner
    self.target_class = target_class
    super(DE_Tune_ML, self).__init__(params_distribution, goal, num_population,
                                     repeats, f, cr, life)

  def evaluate(self):
    # clf = self.learner(self.train_X, self.train_Y, self.test_X, self.test_Y,
    #                    {}, self.tune_goal)
    for n, kwargs in enumerate(self.frontier):
      temp = {}
      for key, val in self.learner.learn({}, **kwargs).iteritems():
        if key in self.target_class:
          temp[key] = val[0]  # value, not list
      self.scores[n] = temp
      # each return value like this{"mean":0.2,"weighted_mean":0.9}

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
