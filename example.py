from ner_pos import ner2pos
import torch
model = ner2pos('./out')

params = list(model.named_parameters())
for n, p in params:
    print(n)
input_ids, input_mask, segment_ids, valid_positions = model.ner_module.preprocess("Steve went to Paris")
input_ids = torch.tensor([input_ids], dtype=torch.long)
input_mask = torch.tensor([input_mask], dtype=torch.long)
segment_ids = torch.tensor([segment_ids], dtype=torch.long)
logits = model(input_ids, input_mask, segment_ids)
active_loss = input_mask.view(-1) == 1
active_logits = logits.view(-1, self.ner_module.model_config["num_labels"])[active_loss]
print(logits.size())