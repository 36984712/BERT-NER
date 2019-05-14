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

python run_ner.py --data_dir=data/ \
--bert_model=bert-base-cased \
--task_name=pos \
--output_dir=out_pos \
--max_seq_length=128 \
--do_train --num_train_epochs 5 \
--do_eval --warmup_proportion=0.4