from transformers import BertPreTrainedModel, BertForSequenceClassification
from transformers import BertConfig, BertTokenizer
from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cross_entropy

ALL_MODELS = sum((tuple(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()) for conf in (BertConfig,)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


class BERTPairwise(BertPreTrainedModel):
    """ A class to create pair wise BERT model.

    Attributes:
        BertPreTrainedModel ():

    Methods:
        forward(input_ids, attention_mask)
        predict_proba(input_)

    """

    def __init__(self, config):
        """ Constructs RankNet object.

        Args:
            config ():

        """
        config.num_labels = 1
        super(BERTPairwise, self).__init__(config)
        self.bert = BertForSequenceClassification(config)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.init_weights()


    def forward(self, input_ids, attention_mask):
        """ .

        Args:
            input_ids (): Document input_ids
            attention_mask (): Document attention_mask

        Returns:
            prob (): pairwise ranking

        """

        outputs = self.bert(input_ids, attention_mask)
        logits = outputs[0]
        return logits

    def predict_proba(self, input_):
        """ .

        Args:
            input_ ():

        Returns:
            confidence (): pairwise ranking score

        """
        return self.model(input_)



    def create_data_loader(self, df_train):
        text_query_tensors = []
        attn_query_tensors = []
        text_pos_tensors = []
        attn_pos_tensors = []
        text_neg_tensors = []
        attn_neg_tensors = []
        tokenizer_config = {
            "max_length": 128,
            "return_attention_mask": True,
            "return_token_type_ids": True,
            "truncation": True,
            "padding": "max_length"
        }

        for i, e in enumerate(df_train.iterrows()):
            tokenized_center = self.tokenizer(e[1]["query"], **tokenizer_config)
            tokenized_pos = self.tokenizer(e[1]["doc1"], **tokenizer_config)
            tokenized_neg = self.tokenizer(e[1]["doc2"], **tokenizer_config)
            text_query_tensors.append(tokenized_center['input_ids'])
            attn_query_tensors.append(tokenized_center['attention_mask'])

            text_pos_tensors.append(tokenized_pos['input_ids'])
            attn_pos_tensors.append(tokenized_pos['attention_mask'])

            text_neg_tensors.append(tokenized_neg['input_ids'])
            attn_neg_tensors.append(tokenized_neg['attention_mask'])

        dataset = TensorDataset(
            torch.tensor(text_query_tensors, dtype=torch.long),
            torch.tensor(attn_query_tensors, dtype=torch.long),
            torch.tensor(text_pos_tensors, dtype=torch.long),
            torch.tensor(attn_pos_tensors, dtype=torch.long),
            torch.tensor(text_neg_tensors, dtype=torch.long),
            torch.tensor(attn_neg_tensors, dtype=torch.long)
        )

        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        return data_loader


def train(data_loader):

    losses = []
    model = BERTPairwise.from_pretrained("bert-base-uncased")
    num_epochs = 50
    optimizer = torch.optim.Adam(model.parameters())
    batch_processed_counter = 0
    for epoch in range(num_epochs):
        print("Epoch: {:2d}/{:2d}   ".format(epoch + 1, num_epochs))
        for i, (query_id, query_att, po_ids, pos_att, neg_id, neg_att) in enumerate(data_loader):
            batch_processed_counter += 1
            pos_output = model.forward(po_ids, pos_att).squeeze(1)
            neg_output = model.forward(neg_id, neg_att).squeeze(1)
            labels = torch.zeros(64, dtype=torch.long)
            loss = cross_entropy(torch.stack((pos_output, neg_output), dim=1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(
                    "Epoch: {:2d}/{:2d}   ".format(epoch + 1, num_epochs),
                    "Batch: {:2d}  ".format(batch_processed_counter),
                    "Batch loss: {:.6f}   ".format(loss.item())
                )