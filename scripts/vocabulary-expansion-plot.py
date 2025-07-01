import matplotlib.pyplot as plt
import json
import numpy as np

all_words = [
    ["perro", "libro", "cielo", "caminar", "mesa", "feliz", "rápido", "bosque", "naranja", "silla"],
    ["사랑", "하늘", "물고기", "친구", "행복", "바람", "산책", "노래", "시간", "별"],
    ["さくら", "ねこ", "すし", "たび", "ゆき", "こころ", "でんしゃ", "やま", "みず", "ひかり"]
]

language_names = ["Spanish", "Korean", "Japanese"]

def get_cumulative_count(data):
    # Find the maximum value to know the range of i
    max_val = max(data)

    # Initialize a list to count occurrences
    count = [0] * (max_val + 1)

    # Count occurrences of each value
    for num in data:
        count[num] += 1

    # Compute cumulative sum
    cumulative_count = []
    total = 0
    for c in count:
        total += c
        cumulative_count.append(total)

    return cumulative_count

fig, ax = plt.subplots(figsize=(4,3), dpi=100)

colors = ["tab:blue", "tab:orange", "tab:green"]

for language_index, words in enumerate(all_words):

    color = colors[language_index]
    language_name = language_names[language_index]

    cumulative_counts = []

    for word in words:
        with open(f"{word}-500.json", "r") as f:
            vocabulary = json.load(f)

        iteration_added = [entry["iteration"] for entry in vocabulary]

        cumulative_count = get_cumulative_count(iteration_added)

        ax.plot(cumulative_count, c=color, alpha=0.1)

        cumulative_counts.append(cumulative_count)

    cumulative_counts = np.array(cumulative_counts)

    ax.plot(np.sum(cumulative_counts, axis=0) / len(words), c=color, label=language_name)
    

ax.plot([0, 500], [1, 5001], c='grey')

ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Vocabulary Size")

ax.set_xlim(0, 500)
ax.set_ylim(0, 5000)

ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("vocabulary-expansion.pdf")
plt.close()