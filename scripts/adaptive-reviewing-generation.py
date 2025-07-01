import os

PATH = '/data2/kenantang/.cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['VLLM_CACHE_ROOT'] = PATH

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# specify the files used for question generation
vocabulary_files = ["perro-500.json", "사랑-500.json", "さくら-500.json"]

# the number of MCQ
mcq_count = 50

# the number of FIB
fib_count = 50

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=2, max_model_len=4096, download_dir='/data/kenantang/llama/hub')

import pandas as pd
import numpy as np
from tqdm import tqdm
import json

sampling_params = SamplingParams(temperature=0, max_tokens=1024, seed=0)

def generate_response_llama(prompt):
    
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    outputs = llm.chat(conversation,
                       sampling_params=sampling_params,
                       use_tqdm=False)


    generated_text = outputs[0].outputs[0].text

    return generated_text

mcq_prompt = """
You are a quiz master creating a quiz based on a knowledge graph of words. Your questions test the ability of a language learner.
Your task is to generate a multiple-choice question (mcq) based on a word and its description.
Your response MUST be a single, valid JSON object. Do not include any other text or explanation.

The JSON format for the question object is the following:
{{"type": "mcq", "question": "Which of the following is a fruit?", "options": ["Apple", "Desk", "Novel", "Creativity"], "answer": "Apple"}}

The questions need to be generated in the same language as the given word. For example, if a given word is a Japanese word, the question itself should be in Japanese.

The answer should not appear in the question.

Here is the word and its description:
Word: {word}
Description: {description}
""".strip()

fib_prompt = """
You are a quiz master creating a quiz based on a knowledge graph of words. Your questions test the ability of a language learner.
Your task is to generate a fill-in-the-blank question based on a word and its description.
Your response MUST be a single, valid JSON object. Do not include any other text or explanation.

The JSON format for the question object is the following:
{{"type": "fill_in_blank", "question": "An _____ is a red and round fruit.", "answer": "apple"}}

The questions need to be generated in the same language as the given word. For example, if a given word is a Japanese word, the question itself should be in Japanese.

The correct answer should be the provided word. The answer should not appear in the question.

Here is the word and its description:
Word: {word}
Description: {description}
""".strip()

outfile = open("mcq-fib.jsonl", "a")

for vocabulary_file in vocabulary_files:
    with open(vocabulary_file, "r") as f:
        vocabulary = json.load(f)

    for i in range(mcq_count):
        word = vocabulary[i]["word"]
        description = vocabulary[i]["description"]

        response = generate_response_llama(mcq_prompt.format(word=word, description=description))
        outfile.write(response)
        outfile.write('\n')
        outfile.flush()


    for i in range(fib_count):
        word = vocabulary[i]["word"]
        description = vocabulary[i]["description"]

        response = generate_response_llama(fib_prompt.format(word=word, description=description))
        outfile.write(response)
        outfile.write('\n')
        outfile.flush()

outfile.close()