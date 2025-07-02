# Evaluation Data of DIY-MKG

This repository contains the scripts and data for the evaluation of the EMNLP 2025 Demo submission "DIY-MKG: An LLM-Based Polyglot Language Learning System".

## Scripts

In the `scripts` folder, 

- `vocabulary-expansion.py` simulates vocabulary expansion in the DIY-MKG interface under a controlled monolingual setting.
- `vocabulary-expansion-plot.py` visualizes the vocabulary expansion results.
- `adaptive-reviewing-generation.py` generates multiple-choice questions and fill-in-the-blank questions using the same prompts in the DIY-MKG interface.
- `adaptive-reviewing-evaluation.py` evaluates the generated questions using `gpt-4.1-2025-04-14`.

To run the scripts, you need to download basic libraries for LLM inference. You also need to manually update the model and data storage paths in the scripts. Please do not hesitate to contact the repository owner if you need additional assistance.

## Data

The `data` folder consists of two subfolders:

- `vocabulary-expansion` includes final vocabularies after 500 rounds of iterative expansion. There are 30 JSON files in total, 10 for each tested language (Spanish, Korean, and Japanese). Each JSON file contains a single array of JSON objects, where `word` is the word, `description` is an LLM-generated description of the word, and `iteration` is the iteration where the word is added into the vocabulary.
- `adaptive-reviewing` includes multiple-choice questions and fill-in-the-blank questions and evaluation results. In `mcq-fib.jsonl`, each line is a multiple-choice question or a fill-in-the-blank question, together with its correct answer. There are 300 questions in total. The evaluation results of the correctness of the question-answer pairs can be found in `mcq-fib-eval.txt`.