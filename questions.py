import transformers
import tensorflow as tf
from transformers import *
import numpy as np


class QuestionsPreparation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare(self, context, question):
        input_ids = self.tokenizer.encode(question, context)
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids)

        input_ids = self.tokenizer.encode(question, context)

        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(self.tokenizer.sep_token_id)

        # Construct the list of 0s and 1s.
        segment_ids = [0]*(sep_index + 1) + [1] * \
            (len(input_ids) - sep_index - 1)
        return tf.constant([input_ids]), tf.constant([segment_ids]), tokens

    def resultDisplay(self, start_pos, end_pos, tokens):
        index_of_sep = tokens.index('[SEP]')
        starting_pos = index_of_sep + 1+ np.argmax(start_pos[index_of_sep+1:-2])
        ending_pos = starting_pos + 1 + np.argmax(end_pos[starting_pos:])
        return self.display_tokens(tokens[starting_pos:ending_pos+1])

    def display_tokens(self, tokens):
        new_tokens = []
        for token in tokens:
            if token[0:2] == "##":
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        return " ".join(new_tokens)
                
