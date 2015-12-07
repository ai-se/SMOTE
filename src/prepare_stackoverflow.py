from __future__ import print_function, division
__author__ = 'rkrsn'
from os import walk
from pdb import set_trace
import numpy as np
import random


def explore(dir='/Users/WeiFu/Github/SMOTE/data/StackExchange/'):
  filenames = ['windowsphone.txt', 'unix.txt', 'softwarerecs.txt',
              'programmers.txt', 'emacs.txt', 'webapps.txt', 'wordpress.txt',
              'webmasters.txt', 'opendata.txt', 'codereview.txt',
              'datascience.txt']
  for afile in filenames:
    datasets = []
    # for (dirpath, dirnames, filenames) in walk(dir): pass
    # # filenames = [f for f in filenames if not 'meta.' in f and 'anime' in
    #  f or 'english' in f]
    # filenames = [f for f in filenames if not 'meta.' in f]
    # sample = np.random.choice(filenames, size=5, replace=False).tolist()
    # set_trace()

    body = {}
    while True:
      sample = np.random.choice(filenames, size=4, replace=False).tolist()
      if afile not in sample:
        sample.append(afile)
        break

    for file in sample:
      b = []
      with open(dir + file) as fp:
        for n, line in enumerate(fp):
          b.append(line.split(' >>> ')[0])
          # writer.write(body+" >>> "+file[:-4]+"\n")
      body.update({file: b})

    tot = 0
    for key in body.keys(): tot += len(body[key])
    # small = sorted(sample, key=lambda F: float(len(body[F])) / tot)
    # set_trace()
    cutoff = tot * random.uniform(0.01, 0.05)
    with open('/Users/WeiFu/Github/SMOTE/data/StackExchange/SE_%s.txt' % (
        afile[:afile.index(".")]), 'w+') as writer:
      for file in sample:
        with open(dir + file) as fp:
          for n, line in enumerate(fp):
            if file == afile:
              if n < cutoff:
                b = line.split(' >>> ')[0]
                writer.write(b + " >>> YES" + "\n")
            else:
              b = line.split(' >>> ')[0]
              writer.write(b + " >>> NO" + "\n")

  # ----- Debug -----
  set_trace()


if __name__ == "__main__":
  explore()
