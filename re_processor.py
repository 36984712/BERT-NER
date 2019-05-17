from run_ner import DataProcessor

import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from relation_ex import relation_extracter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)
label_list = [
    'I-Loc', 'B-Org', 'I-Org', 'B-Other', 'B-Peop', 'I-Peop', 'B-Loc', 'O',
    'I-Other', '[CLS]', '[SEP]', 'X'
]
relation_list = [
    'N', 'Live_In', 'Located_In', 'Work_For', 'OrgBased_In', 'Kill'
]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 guid,
                 text_a,
                 text_b=None,
                 label=None,
                 relation=None,
                 relation_object=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.relation = relation
        self.relation_object = relation_object


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 relation_matrix):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.relation_matrix = relation_matrix
        # self.relation_id = relation_id
        # self.relation_objects = relation_objects


def readfile_relation(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    relation = []
    relation_object = []
    for line in f:
        if len(line) == 0 or line.startswith('#doc') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label, relation, relation_object))
                sentence = []
                label = []
                relation = []
                relation_object = []
            continue
        splits = line.split('\t')
        sentence.append(splits[1])
        # eliminate \n
        label.append(splits[2])
        r = splits[3][1:-1]
        s = r.split(', ')  # some word has more than one relation
        for i in range(len(s)):
            s[i] = s[i][1:-1]  # eliminate "
        relation.append(s)
        ro = splits[4][1:-2]
        so = ro.split(', ')  # some word has more than one relation object
        for i in range(len(so)):
            so[i] = int(so[i])
        relation_object.append(so)
        assert (len(s) == len(so))

    if len(sentence) > 0:
        data.append((sentence, label, relation, relation_object))
        sentence = []
        label = []
        relation = []
        relation_object = []
    return data


class RelationProcessor(DataProcessor):
    """Processor for the CoNLL-2004 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return [
            'I-Loc', 'B-Org', 'I-Org', 'B-Other', 'B-Peop', 'I-Peop', 'B-Loc',
            'O', 'I-Other', '[CLS]', '[SEP]', 'X'
        ]

    def get_relations(self):
        return [
            'N', 'Live_In', 'Located_In', 'Work_For', 'OrgBased_In', 'Kill'
        ]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label, relation,
                relation_object) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            relation = relation
            relation_object = relation_object

            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label,
                             relation=relation,
                             relation_object=relation_object))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile_relation(
            input_file
        )  # num_sentences * (sentence, label, relation, relation_object)


def convert_examples_to_features(examples, label_list, relation_list,
                                 max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # save 0 for the positions less than max length
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    relation_map = {
        r: i
        for i, r in enumerate(relation_list)
    }  # relation[0] is 'N', which means no relation, suitable for padding mask

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        idx_word = dict()
        labellist = example.label
        relationlist = example.relation
        relationobjectlist = example.relation_object
        tokens = []
        labels = []
        relations = []
        relation_objects = []
        idx = 0  # the index of the real tokens
        for i, word in enumerate(textlist):
            idx_word[i] = idx
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            relation_1 = relationlist[i]
            relation_object_1 = relationobjectlist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    relations.append(relation_1)
                    relation_objects.append(relation_object_1)
                else:
                    labels.append("X")
                    relations.append(['N'])
                    relation_objects.append([i + m])
                idx += 1
        for i in range(len(relations)):
            if relations[i] != ['N']:
                for j in range(len(relation_objects[i])):
                    relation_objects[i][j] = idx_word[relation_objects[i][j]]
            else:
                for j in range(len(relation_objects[i])):
                    relation_objects[i][j] = i + j
        # [CLS] is the first one so add 1
        for i in range(len(relations)):
            for j in range(len(relation_objects[i])):
                relation_objects[i][j] += 1
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            relations = relations[0:(max_seq_length - 2)]
            relation_objects = relation_objects[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        relation_ids = []
        nrelation_objects = []

        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        relation_ids.append([relation_map['N']])
        nrelation_objects.append([0])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            current_relation = []
            for r in relations[i]:
                current_relation.append(relation_map[r])
            relation_ids.append(current_relation)
            nrelation_objects.append(relation_objects[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        relation_ids.append([relation_map['N']])
        nrelation_objects.append([len(ntokens) - 1])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            relation_ids.append([0])
            nrelation_objects.append([len(input_ids) - 1])
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(relation_ids) == max_seq_length
        assert len(nrelation_objects) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x)
        #                                             for x in input_ids]))
        #     logger.info("input_mask: %s" %
        #                 " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" %
        #                 " ".join([str(x) for x in segment_ids]))
        #     # logger.info("label: %s (id = %d)" % (example.label, label_ids))
        #     # logger.info("relation: %s" %
        #     #             " ".join([l for l in relation_ids]))
        #     # logger.info("relation_o: %s" %
        #     #             " ".join([l for l in nrelation_objects]))
        #     re = "relation:"
        #     for r in relation_ids:
        #         re += ' ' + str(r)
        #     logger.info(re)
        #     reo = "relation_objects:"
        #     for r in nrelation_objects:
        #         reo += ' ' + str(r)
        #     logger.info(reo)

        # gold_relations = torch.zeros(max_seq_length, max_seq_length,
        #                              len(relation_map))
        gold_relations = [[[0] * len(relation_map)] * max_seq_length] * max_seq_length
        for i, rr in enumerate(relation_ids):
            for j, r in enumerate(rr):
                ob = nrelation_objects[i][j]
                gold_relations[i][ob][r] = 1

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          relation_matrix=gold_relations))
        #                  relation_id=relation_ids,
        #                  relation_objects=nrelation_objects))
    return features


# processor = RelationProcessor()
# re = relation_extracter('out/', 100, 6)
# train_e = processor.get_train_examples('CoNLL04/')
# f = convert_examples_to_features(train_e, label_list, relation_list, 128, re.tokeniser)
# f1 = f[1]
# m = f1.relation_matrix
# print(m[0][0][0])
# print(m[3][3][0])
# print(m[7][28][3])
# print(m[7][12][1])


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--model_dir",
        default='out/',
        type=str,
        required=True,
        help=
        "Dir of the trained Ner model."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help=
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help=
        "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        # n_gpu = torch.cuda.device_count()
        n_gpu = 1  # in case tensor in different gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    processor = RelationProcessor()

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size(
            )
    print("num_train_optimization_steps", num_train_optimization_steps)

    n_classes = len(relation_list)  # num of relations
    transitional_size = 100  # size of transitional layer
    model = relation_extracter('out/', transitional_size, n_classes)

    tokenizer = model.tokeniser
    max_seq_length = model.max_seq_length

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in ['bert'])],
        'weight_decay':
        0.0
    }]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:
        train_features = convert_examples_to_features(train_examples,
                                                      label_list,
                                                      relation_list,
                                                      max_seq_length,
                                                      tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                       dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features],
                                     dtype=torch.long)
        all_relation_matrices = torch.tensor([f.relation_matrix for f in train_features],
                                             dtype=torch.long)
        train_data = TensorDataset(all_input_ids,
                                   all_input_mask,
                                   all_segment_ids,
                                   all_label_ids,
                                   all_relation_matrices)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, relarion_matrices = batch
                relarion_matrices = relarion_matrices.float()
                loss = model(input_ids, segment_ids, input_mask, label_ids, relarion_matrices)["loss"]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model and the associated configuration
        # model_to_save = model.module if hasattr(
        #     model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'ner_re.bin')
        torch.save(model.state_dict(), output_model_file)

    # Load a trained model and config that you have fine-tuned
    else:
        output_model_file = os.path.join(args.output_dir, 'ner_re.bin')
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1
                         or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples,
                                                     label_list,
                                                     relation_list,
                                                     max_seq_length,
                                                     tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],
                                       dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features],
                                     dtype=torch.long)
        all_relation_matrices = torch.tensor([f.relation_matrix for f in eval_features],
                                             dtype=torch.long)
        eval_data = TensorDataset(all_input_ids,
                                  all_input_mask,
                                  all_segment_ids,
                                  all_label_ids,
                                  all_relation_matrices)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data,
                                     sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, relarion_matrices in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            relarion_matrices = relarion_matrices.float()
            relarion_matrices = relarion_matrices.to(device)

            with torch.no_grad():
                output_dict = model(input_ids, segment_ids, input_mask, label_ids, relarion_matrices)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", model.get_metrics())
            writer.write(str(model.get_metrics()))


if __name__ == "__main__":
    main()
