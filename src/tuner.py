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
    for key, val in self.params_distribution:
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
    return candidate

  def evaluate(self):
    raise NotImplementedError("Please implement evaluate")


class DE_Tune_ML(DE):
  def __init__(self, learner,params_distribution,
               goal="F", num_population=60, repeats=60, f=0.75, cr=0.3,
               life=5):
    super(DE_Tune_ML, self).__init__(params_distribution, goal, num_population,
                                     repeats, f, cr, life)
    self.learner = learner


  def evaluate(self):
    F = {}
    # clf = self.learner(self.train_X, self.train_Y, self.test_X, self.test_Y,
    #                    {}, self.tune_goal)
    for n, kwargs in enumerate(self.frontier):
      self.scores[n] = self.learner.learn(**kwargs)
      # each return value like this{"mean":0.2,"weighted_mean":0.9}
