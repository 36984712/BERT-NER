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

python ner_pos.py --data_dir=data_ner_pos/ \
--model_dir=out/ \
--output_dir=out_ner2pos \
--do_train --num_train_epochs 1 \
--do_eval \
--no_cuda