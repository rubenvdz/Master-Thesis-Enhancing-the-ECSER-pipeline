**Enhancing the ECSER pipeline: Evaluating Large
Language Model Classifiers in SE Research**

This repository contains all the code used in the thesis, the results and the source of the pdf. It is structured as follows:

- Code: contains the code used for the mapping study, including the input .bib files containing information about each included paper, the Python file used to parse the bib files (Bibparser.py) and the parsed .bib file and unedited Excel sheet containing the initial list of LLM-related papers.
- Results: contains the file MappingStudyLLM.xlsx with the manually extracted information of the LLM-related papers for the mapping study, including reported metrics and which other steps were taken.
- Thesis: contains the source LaTeX files and auxiliary files used to generate the thesis pdf.
- Replication: contains all the code and results of the replication study on the prediction of Python test results without execution. It includes the data of the original study ("predicting-test-results-gpt-4-main"), the prompts that we developed ("Prompts"), the code that was used to parse the original data ("parse_data.py"), the code used to conduct the replication ("replication.py") and the resulting predictions ("Results").
