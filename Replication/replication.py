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
simple_cases = ['test_positional_only_feature_version','test_january','test_write_simple_dict','test_fileobj_mode','test_basic_formatter']
complex_cases = ['test_AST_objects','test_locale_calendar_formatweekday','test_read_linenum','test_bad_params','test_format_keyword_arguments']

# Load the LLM
llm = Llama.from_pretrained(
 	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
 	filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
    verbose=False,
    logits_all=True,
    n_ctx = 2048
)

# Helper functions
def get_response(prompt, test):
    """
    Get a response from the LLM.
    The response contains the message and logprobs. 
    The message should have the following format (based on the prompts):
    Explanation: <free text>
    Label: <PASS | FAIL>

    Parameters
    ----------
    prompt : prompt with placeholder "<test_case>".
    test : single test case that is given to the LLM

    Returns
    -------
    response : the generated response
    """
    
    current_prompt = prompt.replace("<test_case>", test) # Fill in the actual test in the placeholder
    #print(current_prompt)
    
    response = llm.create_chat_completion(
     	messages = [
    		{
     			"role": "user",
     			"content": current_prompt
    		}
     	],
        logprobs=True,
        top_logprobs=5,
        temperature=0,
    )
    return(response)
    
def evaluate_response(response, verbose=0):
    """
    Take the LLM response and extract the message, label and estimated confidence.

    Parameters
    ----------
    response : llama_cpp response with logprobs.

    Returns
    -------
    message : the generated message of the LLM
    label : classification of PASS or FAIL (or None in case of failure)
    confidence : estimated confidence of classification (between 0 and 1 or None)
    """
    message = response['choices'][0]['message']['content']
    label = response['choices'][0]['logprobs']['content'][-1]['token']
    label = label.replace(" ","") # Remove possible spaces
    confidence = math.exp(response['choices'][0]['logprobs']['content'][-1]['logprob']) # Use the key token probability
    # If the specified format is not followed, the label cannot be automatically extracted
    if (label != "PASS" and label != "FAIL"):
        label = None
        confidence = None
    
    if (verbose):
        print("Message:")
        print(message)
        print("Label:")
        print(label)
        print("Estimated confidence (key token probability):")
        print(confidence)
    
    return message, label, confidence

def get_results(data, prompt, verbose=0, path=None):
    """
    Collect results for all tests in the data set with the given prompt. 

    Parameters
    ----------
    data : Pandas DataFrame with columns suite / name / label / n / test.
    prompt : prompt with placeholder "<test_case>".
    verbose : print all results
    path : optional path to save the results to

    Returns
    -------
    results : Pandas DataFrame that is a copy of the data with added columns for the llm message, predicted label (pred) and confidence.
    """
    results = data.copy()
    results['message'] = ""
    results['pred'] = ""
    results['confidence'] = ""
    for i in data.index:
        test = data['test'][i]
        response = get_response(prompt, test)
        message, pred, confidence = evaluate_response(response, verbose=1)
        results.at[i,'message'] = message
        results.at[i,'pred'] = pred
        results.at[i,'confidence'] = confidence
        
    if path:    
        results.to_csv(path, sep=" ",index=True, header=True,mode="w")
    return results
        

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
results = get_results(val_data[val_data['name'] == "test_AST_objects"], zeroshot, verbose=1,path="Results/test_results.csv")

# response1 = get_response(fewshot, val_data['test'][0])
# response2 = get_response(fewshot, val_data['test'][2])
# evaluate_response(response1, verbose=1)
# evaluate_response(response2, verbose=1)




