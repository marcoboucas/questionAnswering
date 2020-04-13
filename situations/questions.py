"""Question preparation class
"""
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class QuestionsPreparation:
    """Class
    Handle all the pre/post processing of the data
    """
    def __init__(self, transformer):
        self.transformer = transformer

    def prepare(self, context, question):
        """Prepare the data before sending to the algorithm
        """
        input_ids = self.transformer.tokenizer.encode(question, context)
        tokens = self.transformer.tokenizer.convert_ids_to_tokens(
            input_ids)

        input_ids = self.transformer.tokenizer.encode(question, context)

        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(self.transformer.tokenizer.sep_token_id)

        # Construct the list of 0s and 1s.
        segment_ids = [0]*(sep_index + 1) + [1] * \
            (len(input_ids) - sep_index - 1)
        return tf.constant([input_ids]), tf.constant([segment_ids]), tokens

    def result_display(self, start_pos, end_pos, tokens):
        """Display the answer of the question based on
        the results of the algorithm
        """
        index_of_sep = tokens.index('[SEP]')
        starting_pos = index_of_sep + 1+ np.argmax(start_pos[index_of_sep+1:-2])
        ending_pos = starting_pos + 1 + np.argmax(end_pos[starting_pos:])
        return " ".join(self.transformer.remove_hastags(tokens[starting_pos:ending_pos+1]))

    def display_graph(self, start_scores, end_scores, tokens):
        """Graphical explanation
        """
        token_labels = []
        for (i, token) in enumerate(tokens):
            token_labels.append('{:} - {:>2}'.format(token, i))

        # Set up the matplotlib figure
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)


        sns.barplot(x=token_labels, y=start_scores, ci=None, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha="center")
        ax1.grid(True)
        ax1.set_title('Start score')

        sns.barplot(x=token_labels, y=end_scores, ci=None, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha="center")
        ax2.grid(True)
        ax2.set_title('End score')


        plt.show()

    def predict(self, data):
        """Prediction function

        """
        question = data['question']
        context = data['context']

        input_ids, _, tokens = self.prepare(
            context,
            question
        )
        outputs = self.transformer.model({'input_ids': input_ids})
        start_scores, end_scores = outputs
        start_scores = np.array(start_scores).flatten()
        end_scores = np.array(end_scores).flatten()

        answer = self.result_display(start_scores, end_scores, tokens)
        data['answer'] = answer

        return data
