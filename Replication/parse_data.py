# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:38:31 2025

@author: rubenjulian.vdz@gmail.com
"""

import pandas as pd
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import join, getsize
from collections import defaultdict

# This dict will store the test suite, name of the test, true label (fail or pass), number (index) and the actual python test code.
parsed_data = defaultdict(list)

raw_data_dir = "predicting-test-results-gpt-4-main"
# Loop through all subfolders to find the test files and parse information
for path, dirname, filenames in os.walk(raw_data_dir):
    if filenames: 
        for filename in filenames:
            if filename.endswith(".txt"):
                parsed_data["suite"].append(path.split(os.sep)[1])
                parsed_data["name"].append(path.split(os.sep)[2])
                parsed_data["label"].append(filename.split("_")[-2])
                parsed_data["n"].append(filename.split("_")[-1].split(".")[0])
                with open(join(path,filename)) as content:
                    parsed_data["test"].append(content.read().split("===")[0])
                    #print(content.read().split("===")[0])
                #print(filename)
                #print(path.split(os.sep))
                
# Turn the dict into a dataframe
parsed_data = pd.DataFrame(parsed_data)            
# Filter irrelevant data
parsed_data = parsed_data[parsed_data["label"].isin(["pass","fail"])]
parsed_data.to_csv("parsed_data.csv", sep=" ",index=False, header=True,mode="w")
#new_parsed_data = pd.read_csv("parsed_data.csv", sep=" ")
