"""Main file of the project

"""
import os

from transformers_api.load_transformers import TransformerModel
from situations.questions import QuestionsPreparation
from questions.compare import Comparator

def clear():
    """Function that clear the console
    """
    return os.system('clear')  # on Linux System

def main():
    """Main function of the file
    """
    transformer_model = TransformerModel(
        "DistilBert",
        "ForQuestionAnswering",
        "distilbert-base-uncased-distilled-squad"
    )
    question_preparation = QuestionsPreparation(transformer_model)

    comparator = Comparator(transformer_model, 'context.md')

    context = comparator.get_context()
    print(context)
    #clear()
    print('\n'*3)
    question = input('Your question ?\n')

    context = comparator.find_best_paragraph(question, False)
    data = {
        "context" : context,
        "question" : question
    }

    data = question_preparation.predict(data)
    print(data['answer'])

if __name__ == "__main__":
    main()
