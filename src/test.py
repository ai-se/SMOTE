from __future__ import print_function
from mpi4py import MPI

# __author__ = 'WeiFu'
print("hello world")
print("my rank is :%d"%MPI.COMM_WORLD.Get_rank())

