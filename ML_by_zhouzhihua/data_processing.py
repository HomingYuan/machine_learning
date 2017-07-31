#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('data_process.xlsx')

df['low'][df['low']==0] = np.mean(df['low'])
df['high'][df['high']==0] = np.mean(df['high'])

df.to_csv('job_data_process.csv',sep=',')