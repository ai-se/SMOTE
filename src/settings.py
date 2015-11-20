import sys

sys.dont_write_bytecode = True


# __author__ = 'WeiFu'



class o:
  """
  anonymous containers
  """

  def __init__(i, **d): i.has().update(**d)

  def has(i):
    return i.__dict__

  def update(i, d):
    i.has().update(d)
    return i

  def __repr__(i):
    show = [':%s %s' % (k, i.has()[k]) for k in sorted(i.has().keys()) if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'


The = o()


def settings(f=None):
  if f:
    The.__dict__[f.func_name] = f()
  return f


@settings
def cart(**d): return o(
  max_features = None,
  max_depth = None,
  min_samples_split = 2,
  min_samples_leaf = 1,
  # max_leaf_nodes = None,
  ).update(d)


@settings
def rf(**d): return o(
  max_features = "auto",
  min_samples_split = 2,
  min_samples_leaf = 1,
  max_leaf_nodes = None,
  n_estimators = 100
  ).update(d)

@settings
def svm(**d): return o(
  C = 1.0,   # Penalty parameter C of the error term
  epsilon = 0.1,
  kernel = "rbf", ## Specifies the kernel type to be used in the algorithm.
  degree = 3, # Degree of the polynomial kernel function (poly)
  gamma = 0.0, # Kernel coefficient for rbf, poly and sigmoid
  coef0 = 0.0, # Independent term in kernel function
  probability = False, # Whether to enable probability estimates
  shrinking = True, # Whether to use the shrinking heuristic
  tol = 0.001 # Tolerance for stopping criterion.
 ).update(d)



if __name__ == "__main__":
  import pdb
  pdb.set_trace()

  print(The)



