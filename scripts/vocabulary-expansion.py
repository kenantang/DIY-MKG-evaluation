import os

PATH = '/data2/kenantang/.cache'
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
os.environ['VLLM_CACHE_ROOT'] = PATH

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# starting_words = ["perro", "libro", "cielo", "caminar", "mesa", "feliz"]

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# starting_words = ["rápido", "bosque", "naranja", "silla", "사랑", "하늘"]

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# starting_words = ["물고기", "친구", "행복", "바람", "산책", "노래"]

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# starting_words = ["시간", "별", "さくら", "ねこ", "すし", "たび"]

os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
starting_words = ["ゆき", "こころ", "でんしゃ", "やま", "みず", "ひかり"]

number_of_iterations = 500

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

extension_prompt = "Given a word \"{word}\" and its description \"{description}\", suggest 10 new, related words or phrases in the same language that could extend the vocabulary of a language learner. Ensure the list contains words with different parts of speech, such as nouns, verbs, adjectives, adverbs, etc. Your answer must be a single JSON array of strings. Do not include any explanation. For example: [\"apple\", \"pear\", \"orange\"]"

description_prompt = "Generate a short description of the word \"{word}\" in the same language of this word."

for starting_word in starting_words:

    np.random.seed(0)

    starting_description = generate_response_llama(description_prompt.format(word=starting_word))

    vocabulary = [{"word": starting_word, "description": starting_description, "iteration": 0}]
    used_words = []

    for i in range(number_of_iterations):
        chosen_index = np.random.randint(0, len(vocabulary))
        chosen_word = vocabulary[chosen_index]["word"]

        while chosen_word in used_words:
            chosen_index = np.random.randint(0, len(vocabulary))
            chosen_word = vocabulary[chosen_index]["word"]

        chosen_description = vocabulary[chosen_index]["description"]

        used_words.append(chosen_word)

        try:
            new_list = eval(generate_response_llama(extension_prompt.format(word=chosen_word, description=chosen_description)))
        except:
            print(f"Prompt failed on iteration {i}.")

        current_words = [entry["word"] for entry in vocabulary]

        for word in tqdm(new_list):
            if word not in current_words:
                vocabulary.append({
                    "word": word,
                    "description": generate_response_llama(description_prompt.format(word=word)),
                    "iteration": i+1
                })

        print(f"Iteration: {i}, Size: {len(vocabulary)}")

    with open(f"{starting_word}-{number_of_iterations}.json", "w") as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)