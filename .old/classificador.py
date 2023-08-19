import nntplib
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'


class SentimentClassifier(nntplib.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nntplib.Dropout(p=0.3)
        # The last_hidden_state is a sequence of hidden states of the last layer of the model
        self.out = nntplib.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentClassifier(len(class_names))
model = model.to(device)
