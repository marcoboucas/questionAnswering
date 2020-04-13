"""Loading the transformers models and packaging
"""
import transformers

class TransformerModel:
    """Class
    Handling the transformers module and loading
    """

    available_pretrained_weights = {
        'Bert': 'bert-base-uncased',
        'OpenAIGPT': 'openai-gpt',
        'GPT2': 'gpt2',
        'CTRL': "ctrl",
        'XLNet': 'transfo-xl-wt103',
        'XLM': 'xlnet-base-cased',
        'DistilBert': 'distilbert-base-uncased',
        'Roberta': 'roberta-base',
        'XLMRoberta': 'xlm-roberta-base'

    }

    def __init__(
            self,
            model_name="DistilBert",
            model_function="Model",
            pretrained_weights="distilbert-base-uncased"
        ):
        self.model_name = model_name
        self.model_function = model_function
        self.pretrained_weights = pretrained_weights
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()

    def check_if_model(self):
        """Function check if the weights exists
        """
        return self.model_name in self.available_pretrained_weights.keys()

    def get_tokenizer(self):
        """Loading the tokenizer
        """
        tokenizer_name = self.model_name + "Tokenizer"
        return getattr(
            transformers, tokenizer_name).from_pretrained(self.pretrained_weights)

    def get_model(self):
        """Loading the model
        """
        model_name = "TF"+self.model_name+self.model_function
        return getattr(transformers, model_name).from_pretrained(
            self.pretrained_weights)

    def info(self):
        """Display some info
        """
        print(self.model.summary())

    def tokenize(self, text):
        """Tokenize text"""
        input_ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids)
        final = self.remove_hastags(tokens)
        return final

    def remove_hastags(self, tokens):
        """Function that change the tokens to avoid #
        and join the text
        """
        new_tokens = []
        for token in tokens:
            if token not in ['[CLS]', '[SEP]']:
                if token[0:2] == "##":
                    new_tokens[-1] += token[2:]
                else:
                    new_tokens.append(token)
        return new_tokens
