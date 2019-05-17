#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="new"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=127GB

module load python/3.7.1

source ~/venv/ner2emd/bin/activate

python re_processor.py --data_dir=CoNLL04/ \
--model_dir=out/ \
--output_dir=out_re/ \
--max_seq_length=128 \
--do_train --num_train_epochs 1 \
--do_eval --warmup_proportion=0.4