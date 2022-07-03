import torch
import esm

model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()

batch_converter = alphabet.get_batch_converter()

model.eval()

data = [
    ("protein1", "MGMRMMFTVFLLVVLATTVVSFTSGHSGGRKAAAKASNRIALTVRSATCCNYPPCYETYPESCL"),
    ("protein2", "MGMRMMFTVFLLVVLATTVVSFTSGGASGGRKAAAKASNRIALTVRSATCCNYPPCYETYPESCL"),
    ("protein3", "MTDAADLFLMIDEDDVFLMIDEADLFLMIDEDDLFLMIDEDDVFLMIDEA")
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

sequence_representations = []

for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

import matplotlib.pyplot as plt
for (_, seq), attention_contacts in zip(data, results["contacts"]):
    plt.matshow(attention_contacts[: len(seq), : len(seq)])
    plt.title(seq)
    plt.show()
