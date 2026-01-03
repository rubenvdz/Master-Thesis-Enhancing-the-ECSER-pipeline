# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 00:53:23 2025

@author: rubenjulian.vdz@gmail.com
"""

import torch
import pandas as pd
import os
from llama_cpp import Llama


llm = Llama.from_pretrained(
	repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
	filename="Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf",
    verbose=False,
    logits_all=True,
    temperature=0
)

response = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	],
    logprobs=True,
    top_logprobs=5,
)
print(response)