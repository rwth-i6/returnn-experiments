#! /usr/bin/python3

import sys

class ProgressBar(object):

  def __init__(self, msg, fill, width, nSteps):
    self.msg = msg
    self.fill = fill
    self.width = width
    self.nSteps = nSteps
    self.progress_size = nSteps/width

  def restart(self):
    self.step = 0; self.progress = 1
    sys.stdout.write("%s [%s]" % (self.msg, "." * self.width))  
    sys.stdout.flush()
    sys.stdout.write("\b" * (self.width+1)) # return to start of line, after '['

  def next(self):
    self.step += 1
    if self.progress<=self.width and self.step>=self.progress*self.progress_size:
      sys.stdout.write(self.fill)
      sys.stdout.flush()
      self.progress += 1

  def finish(self):
    while self.progress <= self.width:
      sys.stdout.write(self.fill)
      self.progress += 1
    #sys.stdout.write("\n")
    sys.stdout.flush()
