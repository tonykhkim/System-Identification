import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def sliding_windows(x,seq_length):
  x_seq = []
  #y_seq = []
  for i in range(len(x) - seq_length):
    x_seq.append(x[i: i+seq_length])
    #y_seq.append(y[i+sequence_length])

  return x_seq

def sliding_windows_TF(x,y,seq_length):
  x_seq = []
  y_seq = []

  for i in range(len(x) - seq_length+1):
    x_seq.append(x[i: i+seq_length])
    y_seq.append(y[i+seq_length-1])

  return x_seq, y_seq