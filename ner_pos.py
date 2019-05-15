from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig,
                                              BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from run_ner import (InputExample, InputFeatures, readfile_ner, readfile_pos,
                     NerProcessor, PosProcessor, convert_examples_to_features)
from bert import Ner

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class ner2pos(nn.Module):
    def __init__(self, model_dir):
        super(ner2pos, self).__init__()
        self.ner_module = Ner(model_dir)
        self.PosProcessor = PosProcessor()
        self.num_labels = len(self.PosProcessor.get_labels()) + 1
        self.transfer = nn.Linear(self.ner_module.model_config["num_labels"], self.num_labels)

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        # input_ids, input_mask, segment_ids, valid_positions = self.ner_module.preprocess(text)
        # input_ids = torch.tensor([input_ids], dtype=torch.long)
        # input_mask = torch.tensor([input_mask], dtype=torch.long)
        # segment_ids = torch.tensor([segment_ids], dtype=torch.long)
        with torch.no_grad():
            logits_ner = self.ner_module.model(input_ids, segment_ids, input_mask)
        logits_pos = self.transfer(logits_ner)
        if labels is None:
            return logits_pos
        else:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if input_mask is not None:
                active_loss = input_mask.view(-1) == 1
                active_logits = logits_pos.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits_pos.view(-1, self.num_labels),
                                labels.view(-1))
            return loss


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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The model directory where the ner model pretrained."
    )

    # Other parameters
    parser.add_argument(
        "--do_train",
        action='store_true',
        help="Whether to run training.")
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.")
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Total batch size for eval.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
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
        n_gpu = torch.cuda.device_count()
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

    processor = PosProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

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

    # Prepare model
    model = ner2pos(args.model_dir)
    if args.fp16:
        model.half()
    model.to(device)
    model.ner_module.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # param_optimizer = list(model.named_parameters())
    parms = list(model.named_parameters())
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if n_gpu > 1:
        max_length = model.module.ner_module.model_config["max_seq_length"]
        tokenizer = model.module.ner_module.tokenizer
    else:
        max_length = model.ner_module.model_config["max_seq_length"]
        tokenizer = model.ner_module.tokenizer
    if args.do_train:
        train_features = convert_examples_to_features(train_examples,
                                                      label_list,
                                                      max_length,
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
        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_label_ids)
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
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
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
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'ner_pos.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        # with open(output_config_file, 'w') as f:
        #     f.write(model_to_save.config.to_json_string())
        # label_map = {i: label for i, label in enumerate(label_list, 1)}
        # model_config = {
        #     "bert_model": args.bert_model,
        #     "do_lower": args.do_lower_case,
        #     "max_seq_length": args.max_seq_length,
        #     "num_labels": len(label_list) + 1,
        #     "label_map": label_map
        # }
        # json.dump(
        #     model_config,
        #     open(os.path.join(args.output_dir, "model_config.json"), "w"))
        # Load a trained model and config that you have fine-tuned
    else:
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        output_model_file = os.path.join(args.output_dir, 'ner_pos.bin')
        # config = BertConfig(output_config_file)
        # model = BertForTokenClassification(config, num_labels=num_labels)
        model.load_state_dict(torch.load(output_model_file))

    model.to(device)

    if args.do_eval and (args.local_rank == -1
                         or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list,
                                                     max_length,
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
        eval_data = TensorDataset(all_input_ids, all_input_mask,
                                  all_segment_ids, all_label_ids)
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
        for input_ids, input_mask, segment_ids, label_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X":
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])
                    else:
                        temp_1.pop()
                        temp_2.pop()
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
        report = classification_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)


if __name__ == "__main__":
    main()
