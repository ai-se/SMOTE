from __future__ import print_function, division
# __author__ = 'WeiFu'
import pickle
import random
import pdb
import sys
import mpi4py
import os.path
import numpy as np
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import FeatureHasher


class Settings(object):
  def __init__(self, src="../data/StackExchange/academia.txt", corpus=None, tfidf=None):
    self.threshold = 20
    self.data_src = src
    self.processors = 4
    self.data = self.get_data()
    self.corpus = corpus if corpus else self.load_data()
    self.tfidf = tfidf if tfidf else self.make_feature(100, 'tfidf')

  def get_data(self):
    folders = [os.path.join(self.data_src, f) for f in listdir(self.data_src) if
               os.path.isfile(os.path.join(self.data_src, f)) and ".DS" not in f]
    print(folders)
    return folders

  def load_data(self):
    if not self.data_src:
      raise ValueError('data src required!')
    all_label, corpus, used_label = [], [], []
    with open(self.data_src, 'r') as f:
      content = f.read().splitlines()
      for line in content:
        label = line.lower().split(' >>> ')[1].split()[0]
        # ' >>> ' the first term is selected as label.
        all_label.append(label)
        corpus.append([label] + process(line.split(' >>> ')[0]).split())
    label_dist = Counter(all_label)
    for key, val in label_dist:
      if val > self.threshold:
        used_label.append(key)
    used_label.append('others')
    for doc in corpus:
      if doc[0] not in used_label:
        doc[0] = 'others'
    return corpus

  def make_feature(self, num_features=100, method='tfidf'):
    """
    making feature matrix
    :param num: # of features selected
    :param method: "tfidf" and "tf"
    """

    def calculate_tf_idf():
      features, files, tfidf, matrix = {}, {}, {}, []
      for row in self.corpus:
        row_no_label = Counter(row[1:])
        matrix.append(row_no_label)  # keep each row into a matrix
        for key, val in row_no_label.iteritems():
          features[key] = features.get(key, 0) + val
          files[key] = features.get(key, 0) + 1

      for each_feature in files.keys():
        tfidf[each_feature] = (features[each_feature] / sum(features.values())
                               * np.log(len(self.corpus) / files[each_feature]))
      return tfidf, matrix

    def norm(mat):
      """
      l2 normalization. I haven't check it out.
      """
      mat = mat.astype(float)
      for i, row in enumerate(mat):
        nor = np.linalg.norm(row, 2)
        if not nor == 0:
          mat[i] = row / nor
      return mat

    def hash(mat, num_features):
      """
      hashing trick, why need this hash function ????
      """
      hasher = FeatureHasher(num_features)
      X = hasher.transform(mat)
      X = X.toarray()
      return X

    matrix_selected = []
    label = list(zip(*self.corpus))
    if method == 'tfidf':
      tfidf, matrix = calculate_tf_idf()
      features_selected = [pair[0] for pair in sorted(tfidf.items(), key=lambda x: x[1])[-num_features:]]
      for row in matrix:
        matrix_selected.append([row[each] for each in features_selected])
      matrix_selected = np.array(matrix_selected)
      matrix_selected = norm(matrix_selected)
    else:  # tf
      mat = [Counter[row[1:]] for row in self.corpus]
      matrix_selected = hash(mat, num_features)
      matrix_selected = norm(matrix_selected)
    return matrix_selected, label


def process(txt):
  """
  Preprocessing: stemming + stopwords removing
  """
  stemmer = PorterStemmer()
  cached_stopwords = stopwords.words("english")
  return ' '.join(
    [stemmer.stem(word) for word in txt.lower().split() if word not in cached_stopwords and len(word) > 1])


def run(data_src='../data/StackExchange/academia.txt'):
  pass


if __name__ == "__main__":
  # settings().get_data()
  run()
