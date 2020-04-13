"""How to compare two texts"""
import gensim.downloader as api

class Comparator:
    """Class
    Comparing texts
    """
    def __init__(self, transformer, context_file):
        self.transformer = transformer
        self.model = api.load("glove-twitter-25")
        self.tokens, self.paragraphs = self.load_context(context_file)


    def compare_tokens(self, tokens1, tokens2):
        """Function comparing 2 tokens lists
        """
        _tokens1 = [word for word in tokens1 if word in self.model.vocab]
        _tokens2 = [word for word in tokens2 if word in self.model.vocab]
        return self.model.n_similarity(_tokens1, _tokens2)

    def load_context(self, link):
        """Loading context
        """
        with open(link) as file:
            text = file.read()
        paragraphs = text.split('\n')
        tokens = []
        for paragraph in paragraphs:
            tokens.append(self.transformer.tokenize(paragraph))
        return tokens, paragraphs

    def get_context(self):
        """Return the context for display
        """
        return "\n".join(self.paragraphs)

    def find_best_paragraph(self, question, verbose=False):
        """Finding the best paragraph for a question
        """
        question = self.transformer.tokenize(question)
        best_paragraph = 0
        best_similarity = 0
        for i, tokens in enumerate(self.tokens):
            sim = self.compare_tokens(tokens, question)
            if verbose:
                print('='*10)
                print(self.paragraphs[i])
                print(f'Similarity : {sim}')
            if sim > best_similarity:
                best_similarity = sim
                best_paragraph = i
        return self.paragraphs[best_paragraph]
