import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("assets/reviews.csv")

# Teste de toquenizacao
PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# sample_txt = 'Quem conta um conto aumenta um ponto'

# tokens = tokenizer.tokenize(sample_txt)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print(f' Sentence: {sample_txt}')
# print(f'   Tokens: {tokens}')
# print(f'Token IDs: {token_ids}')

# # para cumprir os requisitos de entrada do BERT é
# # preciso ciar vetores de 0s e 1s chamados attention mask,
# # que indicam quais token devem ser considerados como válidos,
# # e adicionar mais três tokens especiais nos textos:

# # [SEP] (102)- Marca o fim de uma frase
# # [CLS] (101)- Deve ser colocado no inicio de cada frase para o BERT saber que trata-se de um problema de classificação.
# # [PAD] (0)- Tokens de valor 0 que devem ser adicionados às sentenças para garantir que todas tenham o mesmo tamanho
# # Isso será feito com o auxílio do método encode_plus()

# encoding = tokenizer.encode_plus(
#     sample_txt,
#     max_length=16,
#     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#     return_token_type_ids=False,
#     pad_to_max_length=True,
#     return_attention_mask=True,
#     return_tensors='pt',  # Return PyTorch tensors
# )

# encoding.keys()

# print(len(encoding['input_ids'][0]))
# encoding['input_ids'][0]

# print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))

token_lens = []

for txt in df.content:
    tokens = tokenizer.encode(txt, max_length=1500)
    token_lens.append(len(tokens))

sns.displot(token_lens, kde=True)
plt.xlabel('Token count')
plt.show()

#max_length=150