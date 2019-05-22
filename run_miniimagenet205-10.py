"""
Train a model on miniImageNet.
"""

import random

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel
from supervised_reptile.miniimagenet import read_dataset
from supervised_reptile.train import train
import os
import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DATA_DIR = '../miniimagenet'
#python -u run_miniimagenet51-01.py --lam_reg 0.1 --shots 1 --inner-batch 10 --inner-iters 8 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 5 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m15-01
#python -u run_miniimagenet201-01.py --lam_reg 0.1 --shots 1 --classes 20 --inner-batch 20 --inner-iters 16 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m201-01
#python -u run_miniimagenet201-01.py --lam_reg 0.1 --shots 1 --classes 20 --inner-batch 20 --inner-iters 16 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m201-01
#python -u run_miniimagenet205-10.py --lam_reg 10.0 --shots 5 --classes 20 --inner-batch 20 --inner-iters 16 --meta-step 1 --meta-batch 5 --meta-iters 100000 --eval-batch 15 --eval-iters 50 --learning-rate 0.001 --meta-step-final 0 --train-shots 15 --checkpoint ckpt_m205-10

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)


    train_set, val_set, test_set = read_dataset(DATA_DIR)
    model = MiniImageNetModel(args.classes, **model_kwargs(args))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        #print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        #print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
