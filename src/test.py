from __future__ import print_function
from mpi4py import MPI

x = range(9)
# __author__ = 'WeiFu'
rank = MPI.COMM_WORLD.Get_rank()
out = x[rank+1]
print("hello world")
print("my rank is :%s"%str(MPI.COMM_WORLD.Get_rank()), "my number:"+str(out))

