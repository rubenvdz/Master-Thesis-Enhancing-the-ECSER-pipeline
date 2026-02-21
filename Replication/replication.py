# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 00:53:23 2025

@author: rubenjulian.vdz@gmail.com
"""

import torch
import pandas as pd
from llama_cpp import Llama, LlamaGrammar
import math
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
)
import numpy as np
from torchmetrics.classification import BinaryCalibrationError
from textattack.augmentation import EmbeddingAugmenter, CharSwapAugmenter
from scipy.stats import friedmanchisquare, binomtest, wilcoxon


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
    n_ctx = 2048,
)

grammar = LlamaGrammar.from_file("Prompts/format.gbnf")

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
        grammar=grammar,
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
    label : classification of "pass" or "fail" (or pd.NA in case of failure)
    confidence : estimated confidence of classification (between 0 and 1 or None)
    """
    message = response['choices'][0]['message']['content']
    label = response['choices'][0]['logprobs']['content'][-1]['token']
    label = label.replace(" ","") # Remove possible spaces
    label = label.lower()
    confidence = math.exp(response['choices'][0]['logprobs']['content'][-1]['logprob']) # Use the key token probability
    # If the specified format is not followed, the label cannot be automatically extracted
    if (label != "pass" and label != "fail"):
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
    n = len(data.index)
    results = data.copy()
    results['message'] = ""
    results['pred'] = ""
    results['confidence'] = ""
    progress = 0
    for i in data.index:
        test = data['test'][i]
        response = get_response(prompt, test)
        message, pred, confidence = evaluate_response(response, verbose=verbose)
        results.at[i,'message'] = message
        results.at[i,'pred'] = pred
        results.at[i,'confidence'] = confidence
        progress += 1
        print(f"{progress} / {n}")
    if path:    
        results.to_csv(path, sep=" ",index=True, header=True,mode="w")
    return results

def evaluate_results(results,print_eval=False,name=""):
    """
    Calculate the following from the results: confusion matrix, precision, recall, specificity, true_accuracy, accuracy, F1, F2, MCC, ROC, AUC, adherence, ECE

    Parameters
    ----------
    results : DataFrame that was obtained from get_results with headers suite / name / label / n  / test / message / pred / confidence
    path : optional path to save the evaluation

    Returns
    -------
    evaluation : dictionary with all the calculated metrics
    """
    n = len(results.index)
    
    results = results[results['pred'].notna()] # Remove failed responses
    n_failed = n - len(results.index)
    y_true = results['label'].values
    y_pred = results['pred'].str.lower()
    
    # Numeric labels (0 and 1), required for some functions
    y_true_binary = (y_true == "fail").astype(int)

    # Probabality of positive class ("fail")
    y_score = np.where(results["pred"] == "fail", results["confidence"], 1.0 - results["confidence"])
    
    # Confusion matrix
    confusion =  confusion_matrix(y_true, y_pred,labels=["fail","pass"])
    tp, fn, fp, tn = confusion.ravel()
    
    # Metrics
    precision = precision_score(y_true, y_pred, pos_label="fail")
    recall = recall_score(y_true, y_pred, pos_label="fail")
    specificity = tn / (tn+fp)
    accuracy = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, pos_label="fail")
    F2 = fbeta_score(y_true, y_pred, beta=2, pos_label="fail")
    MCC = matthews_corrcoef(y_true, y_pred)
    AUC = roc_auc_score(y_true_binary, y_score)
    roc = roc_curve(y_true, y_score, pos_label="fail")
    adherence = (n - n_failed) / n # Percentage of succesful format adherence

    # Calibration
    ece_function = BinaryCalibrationError(n_bins=10, norm="l1")
    ECE = ece_function(torch.tensor(y_score), torch.tensor(y_true_binary)).item()
    
    evaluation = {
        "confusion_matrix": confusion,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "F1": F1,
        "F2": F2,
        "MCC": MCC,
        "AUC": AUC,
        "roc": roc,
        "n_failed": n_failed,
        "adherence": adherence,
        "ECE": ECE,
    }
    if print_eval:
        print_metrics(evaluation,name)
    return evaluation
    
def print_metrics(evaluation, name):
    tp, fn, fp, tn = evaluation['confusion_matrix'].ravel()
    print(f"""{name}
    TP: {tp}
    FP: {fp}
    FN: {fn}
    TN: {tn}
    Precision: {evaluation['precision']}
    Recall: {evaluation['recall']}
    Specificity:   {evaluation['specificity']}
    Accuracy:      {evaluation['accuracy']}
    F1 Score:      {evaluation['F1']}
    F2 Score:      {evaluation['F2']}
    MCC:           {evaluation['MCC']}
    AUC:           {evaluation['AUC']}
          """)
          

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
val_data_fewshot = val_data.drop(index=[21,53,84,158])

with open("Prompts/CVerifier.txt") as f:
    cVerifier = f.read()
    
with open("Prompts/Persona.txt") as f:
    persona = f.read()
    
with open("Prompts/QRefinement.txt") as f:
    qRefinement = f.read()
    
with open("Prompts/QRefinementGPT.txt") as f:
    qRefinementGPT = f.read()


# S4: Prompt comparison.

# Use subset of data for testing code
# small_data = val_data[val_data['n'] == 1].iloc[:4] # Smaller dataset for testing code and prompts
# results_zeroshot = get_results(small_data, zeroshot, verbose=1)
# results_fewshot = get_results(small_data, fewshot, verbose=1)
# results_cVerifier = get_results(small_data, cVerifier, verbose=1)
# results_persona = get_results(small_data, persona, verbose=1)
# results_qRefinement = get_results(small_data, qRefinement, verbose=1)
# results_qRefinementGPT = get_results(small_data, qRefinementGPT, verbose=1)

# Use full validation set
# results_val_zeroshot = get_results(val_data, zeroshot, path="Results/results_val_zeroshot.csv")
# results_val_fewshot = get_results(val_data_fewshot, fewshot, path="Results/results_val_fewshot.csv")
# results_val_cVerifier = get_results(val_data, cVerifier, path="Results/results_val_cVerifier.csv")
# results_val_persona = get_results(val_data, persona, path="Results/results_val_persona.csv")
# results_val_qRefinement = get_results(val_data, qRefinement, path="Results/results_val_qRefinement.csv")
# results_val_qRefinementGPT = get_results(val_data, qRefinementGPT, path="Results/results_val_qRefinementGPT.csv")

# Read saved results
results_val_zeroshot = pd.read_csv("Results/results_val_zeroshot.csv", sep=" ",index_col=0)
results_val_fewshot = pd.read_csv("Results/results_val_fewshot.csv", sep=" ",index_col=0)
results_val_cVerifier = pd.read_csv("Results/results_val_cVerifier.csv", sep=" ",index_col=0)
results_val_persona = pd.read_csv("Results/results_val_persona.csv", sep=" ",index_col=0)
results_val_qRefinement = pd.read_csv("Results/results_val_qRefinement.csv", sep=" ",index_col=0)
results_val_qRefinementGPT = pd.read_csv("Results/results_val_qRefinementGPT.csv", sep=" ",index_col=0)

# Compare results for different prompts on validation set
evaluate_results(results_val_zeroshot,print_eval=True,name="Zeroshot")
evaluate_results(results_val_fewshot,print_eval=True,name="Fewshot")
evaluate_results(results_val_cVerifier,print_eval=True,name="Cognitive Verifier")
evaluate_results(results_val_persona,print_eval=True,name="Persona")
evaluate_results(results_val_qRefinement,print_eval=True,name="Question Refinement")
evaluate_results(results_val_qRefinementGPT,print_eval=True,name="Question Refinement (GPT)")

# S5. Test the models.
# The best results are for persona, so we use this for the final results
# results_test_persona = get_results(test_data, persona, path="Results/results_test_persona.csv")
results_test_persona = pd.read_csv("Results/results_test_persona.csv", sep=" ",index_col=0)
evaluation = evaluate_results(results_test_persona)

# S6. Report the confusion matrix.
print("Confusion Matrix:")
print(evaluation['confusion_matrix'])

# S7. Report Metrics.
simple_results = results_test_persona[results_test_persona['name'].isin(simple_cases)]
complex_results = results_test_persona[results_test_persona['name'].isin(complex_cases)]
suite_results = [results_test_persona[results_test_persona['suite'] == suite] for suite in suites]
# Calculate metrics.
evaluation_simple = evaluate_results(simple_results)
evaluation_complex = evaluate_results(complex_results)
evaluations_suite = [evaluate_results(suite_result) for suite_result in suite_results]

# Print metrics for ALL CASES.
print_metrics(evaluation,"TEST SET ALL")

# Print metrics for simple/complex cases.
print_metrics(evaluation_simple, "TEST SET SIMPLE")
print_metrics(evaluation_complex, "TEST SET COMPLEX")

# Print metrics for each suite.
for i,suite in enumerate(suites):
    print_metrics(evaluations_suite[i],f"TEST SET SUITE {suite}")



# S8. Evaluate calibration, fairness, robustness & sustainability.
# Calibration
print("Calibration:")
print(f"ECE: {evaluation['ECE']}")
# Robustness
# We create two prompts with different prompt attacks: EmbeddingAugmenter and CharSwapAugmenter
# embedding_augmenter = EmbeddingAugmenter(pct_words_to_swap=0.2)
# charswap_augmenter = CharSwapAugmenter(pct_words_to_swap=0.1)
# with open("Prompts/Persona_embedding.txt", "w") as f:
#   f.write(embedding_augmenter.augment(persona)[0])
# with open("Prompts/Persona_charswap.txt", "w") as f:
#   f.write(charswap_augmenter.augment(persona)[0])
# Evaluate the prompts
with open("Prompts/Persona_embedding.txt") as f:
    persona_embedding = f.read()
with open("Prompts/Persona_charswap.txt") as f:
    persona_charswap = f.read()
# results_test_persona_embedding = get_results(test_data, persona_embedding, path="Results/results_test_persona_embedding.csv")
# results_test_persona_charswap = get_results(test_data, persona_charswap, path="Results/results_test_persona_charswap.csv")
results_test_persona_embedding = pd.read_csv("Results/results_test_persona_embedding.csv", sep=" ",index_col=0)
results_test_persona_charswap = pd.read_csv("Results/results_test_persona_charswap.csv", sep=" ",index_col=0)
evaluation_embedding = evaluate_results(results_test_persona_embedding)
evaluation_charswap = evaluate_results(results_test_persona_charswap)
print_metrics(evaluation_embedding,"EMBEDDING ATTACK METRICS")      
print_metrics(evaluation_charswap,"CHARSWAP ATTACK METRICS")      
# Calculate Performance Drop Rate (PDR)
pdr_embedding = 1 - (evaluation_embedding["accuracy"] / evaluation["accuracy"])
pdr_charswap = 1 - (evaluation_charswap["accuracy"] / evaluation["accuracy"])
print(f"PDR Embedding Attack: {pdr_embedding}")
print(f"PDR CharSwap Attack: {pdr_charswap}")



# S9. Analyse overfitting and degradation.
# We calculate degradation:
val_evaluation = evaluate_results(results_val_persona)
print(f"""
    DEGRADATION
    Precision:     {evaluation['precision'] - val_evaluation['precision']}
    Recall:        {evaluation['recall'] - val_evaluation['recall']}
    Specificity:   {evaluation['specificity'] - val_evaluation['specificity']}
    Accuracy:      {evaluation['accuracy'] - val_evaluation['accuracy']}
    F1 Score:      {evaluation['F1'] - val_evaluation['F1']}
    F2 Score:      {evaluation['F2'] - val_evaluation['F2']}
    MCC:           {evaluation['MCC'] - val_evaluation['MCC']}
      """)

# S10: Visualise ROC.
fpr, tpr, thresholds = evaluation['roc']
plt.figure(dpi=1200)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label="ROC curve")
plt.plot([0, 1], [0, 1], linestyle="--", color="orange", label="Random chance")
plt.legend()
plt.savefig('Results/ROC.png', dpi=300)
print(f"AUC: {evaluation['AUC']}")

# S11: Statistical tests.
simple_pred = (simple_results['pred'] == "fail").astype(int).values
complex_pred = (complex_results['pred'] == "fail").astype(int).values
suite_preds = [(result['pred'] == "fail").astype(int).values for result in suite_results]

# Compared to random guessing
preds = (results_test_persona['pred'] == "fail").astype(int).values
n_fail = preds.sum()
n_total = len(preds)
print(binomtest(n_fail, n_total, p=0.5, alternative='two-sided'))

# Between simple and complex
print(wilcoxon(simple_pred,complex_pred))

# Between the suites
print(friedmanchisquare(*suite_preds,))








