#!/bin/bash

python3 train_classification_with_puzzle.py --architecture resnest269  --tag ResNeSt269Puzzle_schedule@optimal_16 --alpha_schedule 0.5 --alpha 4.0  --batch_size 8
# python3 inference_classification.py --architecture resnest50 --tag ResNeSt50Puzzle_schedule_05_a4@optimal_32 --domain train --data_dir '/media/ders/zhangyumin/DATASETS/dataset/newvoc/VOCdevkit/VOC2012/'