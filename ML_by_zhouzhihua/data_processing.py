#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('job_data_process.xlsx')
df[['Low','High']] = df[['Low','High']].astype('float32')
print(df.head())

