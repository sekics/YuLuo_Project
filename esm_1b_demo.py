import torch
import esm

# model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter = alphabet.get_batch_converter()

print("Model Load Over!")
model.eval()

data = [
    ("protein1", "MGMRMMFTVFLLVVLATTVVSFTSGHSGGRKAAAKASNRIALTVRSATCCNYPPCYETYPESCL"),
    ("protein2", "MGMRMMFTVFLLVVLATTVVSFTSGGASGGRKAAAKASNRIALTVRSATCCNYPPCYETYPESCL"),
    ("protein3", "MTDAADLFLMIDEDDVFLMIDEADLFLMIDEDDLFLMIDEDDVFLMIDEA")
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

print("----------------------Batch_labels-----------------------")
print(batch_labels)
print("----------------------Batch_strings----------------------")
print(batch_strs)
print("----------------------Batch_tokens-----------------------")
print(batch_tokens)
print(batch_tokens.size())
# 0 for <bos>
# 1 for <pad>
# 2 for <eos>
# 3 for <unk>
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

print(token_representations.size())
sequence_representations = []

for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))

print(sequence_representations)
print(len(sequence_representations))
print(sequence_representations[0])
print(sequence_representations[0].size())
import matplotlib.pyplot as plt
for (_, seq), attention_contacts in zip(data, results["contacts"]):
    plt.matshow(attention_contacts[: len(seq), : len(seq)])
    plt.title(seq)
    plt.show()
