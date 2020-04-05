from loadTransformer import TransformerModel
from questions import QuestionsPreparation
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="white", context="talk")


TM = TransformerModel(
    "DistilBert",
    "ForQuestionAnswering",
    "distilbert-base-uncased-distilled-squad"
)
QP = QuestionsPreparation(TM.tokenizer)


def clear(): return os.system('cls')  # on Linux System


clear()
question = "What is Tolkien ?"
context = "JRR Tolkien is one of the most famous fantasy writer in the world. He wrote the Lord of the Rings, a story about a humans, dwarves, elves, hobbits (small creatures) and orcs."

print(context)
input_ids, segment_ids, tokens = QP.prepare(
    context,
    question
)


outputs = TM.model({'input_ids': input_ids})
outputs2 = TM.model({'input_ids': input_ids})
start_scores, end_scores = outputs



token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

start_scores = np.array(start_scores).flatten()
end_scores = np.array(end_scores).flatten()
print(question)
print(QP.resultDisplay(start_scores, end_scores, tokens))

# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)


sns.barplot(x=token_labels, y=start_scores, ci=None, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha="center")
ax1.grid(True)
ax1.set_title('Start score')

sns.barplot(x=token_labels, y=end_scores, ci=None, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha="center")
ax2.grid(True)
ax2.set_title('End score')


plt.show()

