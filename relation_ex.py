from bert import Ner
import torch
from torch import nn
import torch.nn.functional as F
from relation_f1_measure import RelationF1Measure


class relation_extracter(nn.Module):
    def __init__(self,
                 model_dir,
                 dim_relation_embed,
                 n_classes,
                 activation: str = "relu"):
        super(relation_extracter, self).__init__()
        self.ner = Ner(model_dir)  # load Ner fine-tuned bert
        self.bert = self.ner.model.bert
        self.config = self.bert.config
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.tokeniser = self.ner.tokenizer
        self.hidden_size = self.config.hidden_size
        self.max_seq_length = self.ner.max_seq_length
        self._activation = activation
        self._n_classes = n_classes

        self._U = nn.Parameter(
            torch.Tensor(self.hidden_size, dim_relation_embed))
        self._W = nn.Parameter(
            torch.Tensor(self.hidden_size, dim_relation_embed))
        self._V = nn.Parameter(torch.Tensor(dim_relation_embed, n_classes))
        self._b = nn.Parameter(torch.Tensor(dim_relation_embed))

        self.init_weights()

        self._relation_metric = RelationF1Measure()

        self._loss_fn = nn.BCEWithLogitsLoss()

    def init_weights(self) -> None:
        """
        Initialization for the weights of the model.
        """
        nn.init.kaiming_normal_(self._U)
        nn.init.kaiming_normal_(self._W)
        nn.init.kaiming_normal_(self._V)

        nn.init.normal_(self._b)

    def multi_class_cross_entropy_loss(self, scores, labels, mask):
        """
        Compute the loss from
        """
        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        # to a suitable format for the relation labels level
        padded_document_length = mask.size(1)
        mask = mask.float()  # Size: n_batches x padded_document_length
        squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0)
        squared_mask = squared_mask.unsqueeze(-1).repeat(
            1, 1, 1, self._n_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        # The scores (and gold labels) are flattened before using
        # the binary cross entropy loss.
        # We thus transform
        flat_size = scores.size()
        scores = scores * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        scores_flat = scores.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)
        labels = labels * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        labels_flat = labels.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)

        loss = self._loss_fn(scores_flat, labels_flat)

        # Amplify the loss to actually see something...
        return 100 * loss

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                relations=None):
        sequence_output, _ = self.bert(input_ids,
                                       token_type_ids,
                                       attention_mask,
                                       output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        ##########################################################
        # Relation Scorer
        # Compute the relation scores
        encoded_text = sequence_output  # Size: batch_size x padded_document_length x bert_output_size(hidden_size)
        left = torch.matmul(
            encoded_text,
            self._U)  # Size: batch_size x padded_document_length x l
        right = torch.matmul(
            encoded_text,
            self._W)  # Size: batch_size x padded_document_length x l

        left = left.permute(1, 0, 2)
        left = left.unsqueeze(3)
        right = right.permute(0, 2, 1)
        right = right.unsqueeze(0)

        B = left + right
        B = B.permute(
            1, 0, 3, 2
        )  # Size: batch_size x padded_document_length x padded_document_length x l

        outer_sum_bias = B + self._b  # Size: batch_size x padded_document_length x padded_document_length x l
        if self._activation == "relu":
            activated_outer_sum_bias = F.relu(outer_sum_bias)
        elif self._activation == "tanh":
            activated_outer_sum_bias = F.tanh(outer_sum_bias)

        relation_scores = torch.matmul(
            activated_outer_sum_bias, self._V
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes
        #################################################################

        # Compute the mask from the text: 1 if there is actually a word in the corresponding sentence, 0 if it has been padded.
        mask = attention_mask
        batch_size, padded_document_length = mask.size()

        relation_sigmoid_scores = torch.sigmoid(
            relation_scores
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes
        # F.sigmoid(relation_scores)

        # predicted_relations[l, i, j, k] == 1 iif we predict a relation k with ARG1==i, ARG2==j in the l-th sentence of the batch
        predicted_relations = torch.round(
            relation_sigmoid_scores
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes

        output_dict = {
            "relation_sigmoid_scores": relation_sigmoid_scores,
            "predicted_relations": predicted_relations,
            "mask": mask,
        }

        if relations is not None:

            # gold_relations[l, i, j, k] == 1 iif we predict a relation k with ARG1==i, ARG2==j in the l-th sentence of the batch
            # when read the file, make the relations as the size of
            # batch_size x padded_document_length x padded_document_length x n_classes
            gold_relations = relations
            # GPU support
            if encoded_text.is_cuda:
                gold_relations = gold_relations.cuda()

            # Compute the loss
            output_dict["loss"] = self.multi_class_cross_entropy_loss(
                scores=relation_scores, labels=gold_relations, mask=mask)

            # Compute the metrics with the predictions.
            self._relation_metric(predictions=predicted_relations,
                                  gold_labels=gold_relations,
                                  mask=mask)

        return output_dict

    def decode(self, output_dict, idx_2_rel_type):
        """
        Decode the predictions
        # idx_2_rel_type = {
        #     0: 'N',
        #     1: 'Live_In',
        #     2: 'Located_In',
        #     3: 'Work_For',
        #     4: 'OrgBased_In',
        #     5: 'Kill'
        # }
        """
        decoded_predictions = []

        predicted_relations = output_dict["predicted_relations"]
        # Size: batch_size x padded_document_length x padded_document_length x n_classes
        # predicted_relations[l, i, j, k] == 1 iif we predict a relation k with ARG1==i, ARG2==j in the l-th sentence of the batch

        for instance_tags in predicted_relations:
            sentence_length = instance_tags.size(0)
            decoded_relations = []

            for arg1, arg2, rel_type_idx in instance_tags.nonzero().data:
                relation = ["*"] * sentence_length
                rel_type = idx_2_rel_type[rel_type_idx.item()]
                relation[arg1] = "ARG1_" + rel_type
                relation[arg2] = "ARG2_" + rel_type
                decoded_relations.append(relation)

            decoded_predictions.append(decoded_relations)

        output_dict["decoded_predictions"] = decoded_predictions

        return output_dict

    def get_metrics(self, reset: bool = False):
        """
        Compute the metrics for relation: precision, recall and f1.
        A relation is considered correct if we can correctly predict the last word of ARG1, the last word of ARG2 and the relation type.
        """
        metric_dict = self._relation_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}  # Dict[str, float]
