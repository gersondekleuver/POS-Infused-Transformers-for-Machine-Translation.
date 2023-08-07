import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', type=str,
                    default='checkpoints/wmt14_transformer_pos_super', help='path to checkpoints')
parser.add_argument('--output_dir', type=str,
                    default='eval', help='path to output dir')
parser.add_argument('--dataset', type=str,  default='wmt14_data',)

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


for i, file in enumerate(os.listdir(checkpoint_dir)):
    if file.endswith('.pt'):

        with open(f"eval/output.txt", "r+") as f:

            f.read()
            f.write(f"checkpoint {str(file)}\n")
            f.write(str(i) + "\n")
            f.close()

            command = f'fairseq_project/utils/pos_generate.py data-bin/{dataset}/pos --gen-subset test --path {checkpoint_dir}/{file} --wandb-project NAT_POS_UD_TEST --validate-interval 1 --ldata data-bin/{dataset}/lang --arch pos_transformer --user-dir ./fairseq_project --criterion length_loss --task pos_translation --optimizer adam --adam-betas "(0.9, 0.98)" --share-decoder-input-output-embed --lr-scheduler inverse_sqrt --warmup-updates 10000 --dropout 0.3 --weight-decay 0.0001  --max-tokens 6000 --label-smoothing 0.1 --save-dir checkpoints/wmt14_transformer_pos --fp16 --warmup-init-lr 1e-07 --tensorboard-logdir eval --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --skip-invalid-size-inputs-valid-test --encoder-layers 12 --decoder-layers 1'

            os.system(f"python {command}")
