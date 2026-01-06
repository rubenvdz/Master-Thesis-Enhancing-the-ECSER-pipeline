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

llm = Llama.from_pretrained(
	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
	filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
    verbose=False,
    logits_all=True,
)

prompt = """The capital of france is Paris
Answer in the following format:
Explanation: <free text>
Label: <TRUE | FALSE>
"""

response = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": prompt
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