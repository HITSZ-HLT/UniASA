import argparse
from collections import defaultdict
import json
import os
import random
import re
from itertools import chain
import sys
sys.path.append('./data_process')
from data_utils import produce_source_target_for_qam
from tqdm import tqdm
from transformers.utils import logging
logger = logging.get_logger(__name__)

stance_id2label = {"1": 'Supporting', "-1": 'Against'}

def extract_quote(s):
    pattern = r"\"([^\"]*)\""
    match = re.search(pattern, s)
    if match:
        extracted_word = match.group(1)
    else:
        extracted_word = "No match found"
    return extracted_word

def get_info(args, orig_data_points):
    info_dict_list = []
    for unique_idx, data_point in enumerate(orig_data_points):
        idx = data_point['doc_id']
        input_text = ' [sent] '.join(data_point['sents'])

        ac_info_dict_dict = {}
        for tup in data_point['labels']:
            if tup[0] not in ac_info_dict_dict:
                ac_info_dict_dict[tup[0]] = {
                    'ordered_id': tup[0],
                    'type': stance_id2label[tup[2]],
                    'text': data_point['sents'][tup[0]]
                }
        ac_info_dict_list = sorted(list(ac_info_dict_dict.values()), key=lambda _: _['ordered_id'])
            
        
        ar_info_dict_list = []
        for tup in data_point['labels']:
            ar_info_dict_list.append({
                'src_ordered_id': tup[0],
                'tgt_ordered_id': tup[1],
                'type': args.ar_type_2_prompt[tup[3]]
            })
        
        info_dict_list.append({
            'idx': unique_idx,
            'ac_info_dict_list': ac_info_dict_list,
            'ar_info_dict_list': ar_info_dict_list,
            'input_text': input_text,
            'topic': data_point['topic']
        })
    return info_dict_list

def preprocess_qam(args):
    print(f'seed: {args.seed}')
    orig_data_path = 'input/orig_datasets/QAM'

    with open(os.path.join(orig_data_path, 'train.json'), 'r') as f:
        orig_train_data_points = json.load(f)
    with open(os.path.join(orig_data_path, 'dev.json'), 'r') as f:
        orig_dev_data_points = json.load(f)
    with open(os.path.join(orig_data_path, 'test.json'), 'r') as f:
        orig_test_data_points = json.load(f)
    
    num_test_data_points = len(orig_test_data_points)
    num_dev_data_points = len(orig_dev_data_points)
    all_data_points = orig_train_data_points + orig_dev_data_points + orig_test_data_points
    all_data_info_dict_list = get_info(args, all_data_points)


    # print the max number of acs in all essays
    max_ac_num = max([len(_['ac_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'max_ac_num: {max_ac_num}') # 48
    # print the total number of acs
    total_num_ac = sum([len(_['ac_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'total_num_ac: {total_num_ac}') # should be 25563
    # print the total number of ars
    total_num_ar = sum([len(_['ar_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'total_num_ar: {total_num_ar}') # should be 25563

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # all_data_info_dict_list = post_process(all_data_info_dict_list)

    data_points_list = []
    for info_dict in tqdm(all_data_info_dict_list):
        if args.use_mv == False:
            data_points = produce_source_target_for_qam(args, info_dict, view_types=['[ASA]'])
        else:
            data_points = produce_source_target_for_qam(args, info_dict, view_types=['[ACI]', '[ARI]', '[ASA]'])
        data_points_list.append(data_points)

    train_data_points_list = data_points_list[:-(num_dev_data_points + num_test_data_points)]
    dev_data_points_list = data_points_list[-(num_dev_data_points + num_test_data_points):-num_test_data_points]
    test_data_points_list = data_points_list[-num_test_data_points:]

    # reduce train dev test data_points_list into *_data_points
    train_data_points = list(chain.from_iterable(train_data_points_list))
    dev_data_points = list(chain.from_iterable(dev_data_points_list))
    test_data_points = list(chain.from_iterable(test_data_points_list))
    
    # Save the splitted dataset to JSON files
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    datasets = {
        'train': train_data_points,
        'validation': dev_data_points,
        'test': test_data_points
    }

    for filename, data in datasets.items():
        with open(os.path.join(save_dir, f'{filename}.json'), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['qam'], default='qam')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev_ratio', type=float, default=0.1)

    parser.add_argument('--save_dir', type=str, default='input')
    parser.add_argument('--use_mv', action='store_true', help='use multi-view learning')
    args = parser.parse_args()

    args.ar_type_2_prompt = {
        'Expert': 'is supported by the "Expert" evidence',
        'Research': 'is supported by the "Research" evidence',
        'Case': 'is supported by the "Case" evidence',
        'Explanation': 'is supported by the "Explanation" evidence',
        'Others': 'is supported by the "Others" evidence'
    }
    args.ar_prompt_2_type = {v: k for k, v in args.ar_type_2_prompt.items()}

    # setup_seed(args.seed)

    # 3407 1996 43 1024 2049
    # python data_process/data_preprocess_cdcp.py --dataset_type cdcp --view_type mv
    # python -m debugpy --listen 5432 --wait-for-client data_process/data_preprocess_cdcp.py --dataset_type cdcp --view_type mv
    
    if args.dataset_type == 'qam':
        args.save_dir = os.path.join(args.save_dir, 'qam')
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_mv:
        args.save_dir = os.path.join(args.save_dir, 'mv')
    else:
        args.save_dir = os.path.join(args.save_dir, 'sv')

    preprocess_qam(args)
