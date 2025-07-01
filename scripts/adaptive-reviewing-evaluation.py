import json
from openai import OpenAI
from tqdm import tqdm

model_name = "gpt-4.1-2025-04-14"


client = OpenAI()

input_filename = "mcq-fib.jsonl"
questions = []
with open(input_filename, "r", encoding="utf-8") as f:
    for line in f:
        questions.append(line.strip())

evaluation_prompt = """
Please evaluate if the answer to the question is correct. Only output YES or NO without any explanation.
{question_object}
""".strip()

outfile = open("mcq-fib-eval.txt", "a")

for question in questions:

    response = client.chat.completions.create(
        model=model_name,  # Replace with appropriate model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt.format(question_object=question)}
        ],
        temperature=0.0
    )

    outfile.write(response.choices[0].message.content.strip())
    outfile.write("\n")
    outfile.flush()

outfile.close()