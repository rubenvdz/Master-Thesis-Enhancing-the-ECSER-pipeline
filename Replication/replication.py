# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 00:53:23 2025

@author: rubenjulian.vdz@gmail.com
"""

import torch
import pandas as pd
import os
from llama_cpp import Llama
import math

# Get the data set
data = pd.read_csv("parsed_data.csv", sep=" ",index_col=0)
suites = data['suite'].unique()
test_cases = data['name'].unique()
simples_cases = ['test_positional_only_feature_version','test_january','test_write_simple_dict','test_fileobj_mode','test_basic_formatter']
complex_cases = ['test_AST_objects','test_locale_calendar_formatweekday','test_read_linenum','test_bad_params','test_format_keyword_arguments']


# S1: Select an Evaluation Method and Split the Data.
# We use the holdout method and take 50% of tests for each test case while keeping an even PASS/FAIL split
# We can do this simply using the 'n' column since it numbers each test case and label from 1 to 10
val_data = data[data['n'] <= 5]
test_data = data[data['n'] > 5]

# Test the split criteria:
assert(len(val_data) == len(test_data))
assert(val_data['label'].value_counts()['fail'] == val_data['label'].value_counts()['pass']) # Class balance
assert(test_data['label'].value_counts()['fail'] == test_data['label'].value_counts()['pass']) # Class balance
assert(all(val_data['name'].value_counts() == 10) & all(test_data['name'].value_counts() == 10)) # All cases are represented equally

# S3: Design prompts.
# We define multiple prompts that can be compared on the validation data.

with open("Prompts/Zeroshot.txt") as f:
    zeroshot = f.read()

# Examples: test_positional_only_feature_version_fail_1, test_january_pass_1, test_read_linenum_fail_1, test_fileobj_mode_pass_1
# Indices: 21,53,84,158
with open("Prompts/Fewshot.txt") as f:
    fewshot = f.read()
    
# Remove few-shot examples from val set:
val_data = val_data.drop(index=[21,53,84,158])

# S4: Prompt comparison.
# Start by comparing zero shot and few shot prompts

llm = Llama.from_pretrained(
 	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
 	filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
    verbose=False,
    logits_all=True,
)

response = llm.create_chat_completion(
 	messages = [
		{
 			"role": "user",
 			"content": zeroshot
		}
 	],
    logprobs=True,
    top_logprobs=5,
    temperature=0,
)
print("Response:")
print(response['choices'][0]['message']['content'])
print("Label:")
print(response['choices'][0]['logprobs']['content'][-1]['token'])
print("Estimated confidence (key token probability): ")
print(math.exp(response['choices'][0]['logprobs']['content'][-1]['logprob']))


