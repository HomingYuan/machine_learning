# -*- coding: utf-8 -*

import pandas as pd
import math
import numpy as np
import random
import matplotlib.pyplot as plt

class classifier(object):

    def __init__(self,dataSet):
        self.dataSet = dataSet

    def dist(num1, num2):
        t = (num1 - num2) ** 2
        return float(t ** 0.5)

    def mean_dist(self):
        d = {}
        a = np.array(self.dataSet)






