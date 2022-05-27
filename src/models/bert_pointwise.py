import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERT(nn.Module):
    """ A class to create pair wise RankNet models.

    Attributes:
        num_features ():

    Methods:
        forward(input1, input2)
        predict_proba(input_)

    """

    def __init__(self):
        """ Constructs RankNet object. """
        super(BERT, self).__init__()
        pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco"

        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
        self.bert_model = AutoModel.from_pretrained(pre_trained_model_name)
        self.model = lambda input:(
        self.bert_model(**self.tokenizer(input, return_tensors="pt"))[0][:, 0, :].squeeze(0))



    def forward(self, input):
        """ Trains model on inputs.

        Args:
            input (): Document 1 features

        Returns:
            prob (): pointwise ranking

        """
        #output1 = self.model(input)
        pass # skip training, model is already trained on the dataset


    def predict_proba(self, passage, query):
        """ .

        Args:
            passage ():
            query ():

        Returns:
            confidence (): pointwise ranking score

        """
        passage_encoded = self.model(passage)
        query_encoded = self.model(query)
        return query_encoded.dot(passage_encoded)
