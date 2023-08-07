import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', type=str,
                    default='checkpoints/POSDP_IWSLT/checkpoint_best.pt', help='path to checkpoints')
parser.add_argument('--output_dir', type=str,
                    default='eval', help='path to output dir')
parser.add_argument('--dataset', type=str,  default='wmt14_en_de',)

args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
output_dir = args.output_dir
dataset = args.dataset


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create evaluation.txt file
try:
    with open(f"eval/output.txt", "w+") as f:
        f.write("NEW RUN\n")
        f.close()
except:
    pass

# pentalties = np.linspace(0, 0.1, 11)

pentalties = [0.05]

for penalty in pentalties:
    with open(f"eval/output.txt", "r+") as f:
        f.read()
        f.write(f"penalty: {penalty}\n")
        f.close()

    command = f'fairseq_project/utils/POSDP_generate.py data-bin/wmt14_en_de/lang --penalty {penalty} --user-dir ./fairseq_project --gen-subset test --arch POSDP_transformer --criterion POSDP_criterion --task POSDP_task --path checkpoints/POSDP/checkpoint_best.pt --iter-decode-max-iter 9 --iter-decode-eos-penalty 0 --beam 4 --remove-bpe --print-step --batch-size 400'

    os.system(f"python {command}")
