import os
import nltk
import json
import pandas as pd
import random
import argparse
import re
from collections import defaultdict
from copy import deepcopy
from data_utils import produce_source_target
from tqdm import tqdm
import numpy as np
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def check_data_info(all_data_info_dict_list):
    ac_info_lists = [x['ac_info_dict_list'] for x in all_data_info_dict_list]
    ar_info_lists = [x['ar_info_dict_list'] for x in all_data_info_dict_list]

    max_ac_num = max(map(len, ac_info_lists))
    total_num_ac = sum(map(len, ac_info_lists))
    total_num_ar = sum(map(len, ar_info_lists))

    print(f'max_ac_num: {max_ac_num}') # 28
    print(f'total_num_ac: {total_num_ac}') # should be 6089
    print(f'total_num_ar: {total_num_ar}') # should be 3832

def get_spans(orig_essay, ac_text_list):
    spans = []
    start, end = 0, 0
    valid_ac_flags = []
    for ac_text in ac_text_list:
        start = orig_essay.find(ac_text, end)
        if start != -1:
            end = start + len(ac_text)
            spans.append([start, end])
            valid_ac_flags.append(True)
        else:
            valid_ac_flags.append(False)
    # valid_ac_flags means whether the ac_text is valid in orig_essay
    return spans, valid_ac_flags

def post_process_text(s):
    # replace all non-space whitespace characters with space
    s = re.sub(r'[^\S ]', ' ', s)
    # replace multiple spaces with one space
    s = re.sub(r' +', ' ', s).strip()
    # using regex to replace the space before punctuation
    s = re.sub(r'\s([,.!?;:])', r'\1', s)
    return s

def post_process(orig_all_data_info_dict_list):
    # check consecutive space tokens
    for data_info_dict in orig_all_data_info_dict_list:
        data_info_dict['input_text'] = data_info_dict['input_text'].replace('\n', ' [NEW_LINE] ')
        # replace multiple spaces with one space
        data_info_dict['input_text'] = post_process_text(data_info_dict['input_text'])
        ac_text_list = [post_process_text(ac['text']) for ac in data_info_dict['ac_info_dict_list']]
        spans, valid_ac_flags = get_spans(data_info_dict['input_text'], ac_text_list)
        assert len(spans) == len(ac_text_list) == sum(valid_ac_flags)

        for span, ac_text, ac_info_dict in zip(spans, ac_text_list, data_info_dict['ac_info_dict_list']):
            # if ac_text != ac_info_dict['text']:
            #     print("\nbefore vs after post process")
            #     print(ac_text)
            #     print(ac_info_dict['text'])
            ac_info_dict['start'] = span[0]
            ac_info_dict['end'] = span[1]
            ac_info_dict['text'] = ac_text
    
    return orig_all_data_info_dict_list


def split_essay_into_paragraph(data_info_dict_list):
    # split the essay into paragraphs
    for data_info_dict in data_info_dict_list:
        essay = data_info_dict['input_text']
        topic = data_info_dict['topic']
        ac_info_dict_list = data_info_dict['ac_info_dict_list']
        ar_info_dict_list = data_info_dict['ar_info_dict_list']
        idx = data_info_dict['idx']
        assert len(essay.strip()) == len(essay)

        # find the position of each '\n' in the essay by using the find() function
        break_pos_list = []
        start_pos = 0
        while True:
            pos = essay.find(' [NEW_LINE] ', start_pos)
            if pos == -1:
                break
            break_pos_list.append(pos)
            start_pos = pos + 1
        assert len(break_pos_list) > 2
        break_pos_list.append(len(essay))

        # group the acs by the paragraph, make sure that each ac is in the correct paragraph
        ac_added_flag = [False] * len(ac_info_dict_list)
        paragraph_info_dict_list = []
        paragraph_start_pos = 0
        for p_id, pos in enumerate(break_pos_list):
            ac_group = []
            this_paragraph = essay[paragraph_start_pos:pos]
            for ac_idx, ac in enumerate(ac_info_dict_list):
                if ac['start'] >= paragraph_start_pos and ac['start'] < pos \
                    and ac['end'] >= paragraph_start_pos and ac['end'] <= pos:
                    new_ac = deepcopy(ac)
                    # new_ac['paragraph'] = this_paragraph
                    # change the start and end pos of acs accordingly
                    new_ac['start'] = new_ac['start'] - paragraph_start_pos
                    new_ac['end'] = new_ac['end'] - paragraph_start_pos
                    assert this_paragraph[new_ac['start']:new_ac['end']] == new_ac['text']
                    ac_group.append(new_ac)
                    ac_added_flag[ac_idx] = True
            paragraph_start_pos = pos + 1

            # update the paragraph level ordered id
            for ordered_id, ac in enumerate(ac_group):
                ac['ordered_id'] = ordered_id

            paragraph_info_dict_list.append({
                'idx': idx,
                'ac_info_dict_list': ac_group,
                'ar_info_dict_list': [],
                'input_text': this_paragraph,
                'topic': topic,
                'paragraph_idx': p_id,
            })
        
        ar_added_flag = [False] * len(ar_info_dict_list)
        for paragraph_info_dict in paragraph_info_dict_list:
            ann_id_to_ordered_id = {ac['ann_id']: ac['ordered_id'] for ac in paragraph_info_dict['ac_info_dict_list']}
            for ar_idx, ar_info_dict in enumerate(ar_info_dict_list):
                if ar_info_dict['src_ann_id'] in ann_id_to_ordered_id and ar_info_dict['tgt_ann_id'] in ann_id_to_ordered_id:
                    new_ar = deepcopy(ar_info_dict)
                    # update to the ordered id in the paragraph
                    new_ar['src_ordered_id'] = ann_id_to_ordered_id[ar_info_dict['src_ann_id']]
                    new_ar['tgt_ordered_id'] = ann_id_to_ordered_id[ar_info_dict['tgt_ann_id']]
                    paragraph_info_dict['ar_info_dict_list'].append(new_ar)
                    ar_added_flag[ar_idx] = True

            # update the paragraph level ordered id
            for ordered_id, ar in enumerate(paragraph_info_dict['ar_info_dict_list']):
                ar['ordered_id'] = ordered_id

        assert sum(ar_added_flag) == len(ar_info_dict_list)
        assert sum(ac_added_flag) == len(ac_info_dict_list)
        assert len(ac_info_dict_list) == sum([len(_['ac_info_dict_list']) for _ in paragraph_info_dict_list])

        data_info_dict['paragraph_info_dict_list'] = paragraph_info_dict_list
    
    return data_info_dict_list

def preprocess_essay_level(args):
    print(f'Processing: seed-{args.seed}')

    aaec_file_path = 'input/orig_datasets/ArgumentAnnotatedEssays-2.0-processed/brat-project-final'
    orig_all_data_info_dict_list = parse_aaec(args, aaec_file_path)
    check_data_info(orig_all_data_info_dict_list)
    all_data_info_dict_list = post_process(orig_all_data_info_dict_list)
    
    data_points_list = []
    for info_dict in tqdm(all_data_info_dict_list):
        if args.use_mv == False:
            data_points = produce_source_target(args, info_dict, view_types=['[ASA]'])
        else:
            data_points = produce_source_target(args, info_dict, view_types=['[ACI]', '[ARI]', '[ASA]'])
        data_points_list.extend(data_points)

    split_and_save_data(args, data_points_list)

def preprocess_paragraph_level(args):
    print(f'Processing: seed-{args.seed}')

    aaec_file_path = 'input/orig_datasets/ArgumentAnnotatedEssays-2.0-processed/brat-project-final/'
    orig_all_data_info_dict_list = parse_aaec(args, aaec_file_path)
    # check_data_info(all_data_info_dict_list)
    all_data_info_dict_list = post_process(orig_all_data_info_dict_list)

    # split the essay into paragraphs
    # add an 'paragraph_info_dict_list' key for each dict in essay_info_dict_list
    all_data_info_dict_list = split_essay_into_paragraph(all_data_info_dict_list)

    data_points_list = []
    for data_info_dict in tqdm(all_data_info_dict_list):
        paragraph_info_dict_list = data_info_dict['paragraph_info_dict_list']

        for info_dict in paragraph_info_dict_list:
            if args.use_mv == False:
                data_points = produce_source_target(args, info_dict, view_types=['[ASA]'])
            else:
                data_points = produce_source_target(args, info_dict, view_types=['[ACI]', '[ARI]', '[ASA]'])
            # add paragraph id
            for data_point in data_points:
                data_point['idx'] = f"{data_point['idx']}_{info_dict['paragraph_idx']}"
            data_points_list.extend(data_points)

    split_and_save_data(args, data_points_list, 'paragraph')

def split_and_save_data(args, data_points_list, level='essay'):
    # Get the ID lists for train, dev and test sets
    train_essay_ids, dev_essay_ids, test_essay_ids, train_dev_essay_ids = \
        get_train_dev_test_ids(args.seed, args.dev_ratio)

    # Split the data points into respective sets based on their IDs
    train_dev_data_points, train_data_points, dev_data_points, test_data_points = [], [], [], []
    for data in data_points_list:
        if level == 'essay':
            essay_id = data['idx']
        elif level == 'paragraph':
            essay_id = int(data['idx'].split('_')[0])
        else:
            raise ValueError(f'Invalid level: {level}')
    
        if essay_id in train_essay_ids:
            train_data_points.append(data)
            train_dev_data_points.append(data)
        elif essay_id in dev_essay_ids:
            dev_data_points.append(data)
            train_dev_data_points.append(data)
        elif essay_id in test_essay_ids:
            test_data_points.append(data)
        else:
            raise ValueError(f'essay_id {essay_id} not in train_essay_ids, dev_essay_ids, test_essay_ids')

    # Create the directory to save the datasets if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Define a dictionary to hold the datasets
    datasets = {
        'train_validation': train_dev_data_points,
        'train': train_data_points,
        'validation': dev_data_points,
        'test': test_data_points
    }

    # Save each dataset to a JSON file
    for filename, data in datasets.items():
        with open(os.path.join(args.save_dir, f'{filename}.json'), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def get_train_dev_test_ids(seed, dev_ratio):
    split_df = pd.read_csv('input/orig_datasets/ArgumentAnnotatedEssays-2.0-processed/train-test-split.csv', sep=';')

    # get all the essay ids in the train set
    train_essay_ids = [int(_[-3:]) for _ in split_df[split_df['SET'] == 'TRAIN']['ID']]
    test_essay_ids = [int(_[-3:]) for _ in split_df[split_df['SET'] == 'TEST']['ID']]

    orig_train_essay_ids = deepcopy(train_essay_ids)

    # split dev set from train set
    random.seed(seed)
    random.shuffle(train_essay_ids)
    dev_num = int(len(train_essay_ids) * dev_ratio)
    dev_essay_ids = train_essay_ids[-dev_num:]
    train_essay_ids = train_essay_ids[:-dev_num]

    train_essay_ids_set = set(train_essay_ids)
    dev_essay_ids_set = set(dev_essay_ids)
    test_essay_ids_set = set(test_essay_ids)
    orig_train_essay_ids_set = set(orig_train_essay_ids)

    assert len(train_essay_ids_set) == len(train_essay_ids)
    assert len(dev_essay_ids_set) == len(dev_essay_ids)
    assert len(test_essay_ids_set) == len(test_essay_ids)
    assert len(orig_train_essay_ids_set) == len(orig_train_essay_ids)

    return train_essay_ids, dev_essay_ids, test_essay_ids, orig_train_essay_ids


def parse_aaec(args, essay_path):
    ac_type_list = ['MajorClaim', 'Claim:For', 'Claim:Against', 'Premise']

    essay_ac_info_list = []
    for essay_id in range(1, 403):
        raw_essay = open(os.path.join(essay_path, 'essay{:03d}.txt'.format(essay_id)), 'r').read()
        raw_ann = open(os.path.join(essay_path, 'essay{:03d}.ann'.format(essay_id)), 'r').read()

        ann_list = raw_ann.split('\n')
        ac_ann_list = [ann for ann in ann_list if ann.startswith('T')]
        ar_ann_list = [ann for ann in ann_list if ann.startswith('R')]
        claim_stance_ann_list = [ann for ann in ann_list if ann.startswith('A')]
        ac_info_dict = {} # key: ac id in `ann_list`, like 'T1', value: ac_info, which is a dict

        # get the raw ac info
        for ann in ac_ann_list:
            ann_ac_id = ann.split('\t')[0]
            ac_info = ann.split('\t')[1]
            ac_type = ac_info.split(' ')[0]
            ac_start = int(ac_info.split(' ')[1])
            ac_end = int(ac_info.split(' ')[2])
            ac_text = ann.split('\t')[2]
            assert ac_text == raw_essay[ac_start:ac_end]
            
            ac_info = {
                'ann_id': ann_ac_id,
                'type': ac_type,
                'start': ac_start,
                'end': ac_end,
                'text': ac_text
            }
            assert ac_type in ['MajorClaim', 'Claim', 'Premise']
            ac_info_dict[ann_ac_id] = ac_info

        ar_info_dict = {} # key: ar id in `ann_list`, like 'R1', value: ar_info, which is a dict
        # get the raw ar info
        for ann in ar_ann_list:
            ann_ar_id, raw_rel = ann.split('\t')[:2]
            ar_type, ar_arg1, ar_arg2 = raw_rel.split(' ')
            ar_arg1_id = ar_arg1.split(':')[1]
            ar_arg2_id = ar_arg2.split(':')[1]
            assert ar_arg1_id.startswith('T') and ar_arg2_id.startswith('T')
            assert ar_type in ['attacks', 'supports']
            assert ar_arg1_id in ac_info_dict and ar_arg2_id in ac_info_dict
            assert ac_info_dict[ar_arg1_id]['type'] == 'Premise'
            assert ac_info_dict[ar_arg2_id]['type'] in ['Premise', 'Claim']

            ar_info = {
                'ann_id': ann_ar_id,
                'type': args.ar_type_2_prompt[ar_type],
                'src_ann_id': ar_arg1_id,
                'tgt_ann_id': ar_arg2_id
            }
            ar_info_dict[ann_ar_id] = ar_info

        # acs are sorted by their start pos
        ordered_ac_info_list = sorted(ac_info_dict.values(), key=lambda x: x['start'])
        ann_ac_id2ordered_ac_id = {ac['ann_id']: i for i, ac in enumerate(ordered_ac_info_list)}
        ordered_ac_id2ann_ac_id = {i: ac['ann_id'] for i, ac in enumerate(ordered_ac_info_list)}
        for ac in ordered_ac_info_list:
            ac['ordered_id'] = ann_ac_id2ordered_ac_id[ac['ann_id']]
        
        ar_info_list = list(ar_info_dict.values())
        for ar in ar_info_list:
            ar['src_ordered_id'] = ann_ac_id2ordered_ac_id[ar['src_ann_id']]
            ar['tgt_ordered_id'] = ann_ac_id2ordered_ac_id[ar['tgt_ann_id']]
        # ars are sorted by the id of their src ac
        ordered_ar_info_list = sorted(ar_info_list, key=lambda x: x['src_ordered_id'])
        ann_ar_id2ordered_ar_id = {ar['ann_id']: i for i, ar in enumerate(ordered_ar_info_list)}
        ordered_ar_id2ann_ar_id = {i: ar['ann_id'] for i, ar in enumerate(ordered_ar_info_list)}
        for ar in ordered_ar_info_list:
            ar['ordered_id'] = ann_ar_id2ordered_ar_id[ar['ann_id']]

        # check if there exists two consecutive acs
        for i in range(len(ordered_ac_info_list) - 1):
            assert ordered_ac_info_list[i]['end'] < ordered_ac_info_list[i + 1]['start']

        # get the ordered ac and ar info list for later use
        ordered_ac_info_list
        ordered_ar_info_list

        # clean the raw_essay
        assert raw_essay[0] != ' '
        assert raw_essay[0] != '\n'
        raw_essay = raw_essay.strip()
        if '\n\n' in raw_essay:
            raw_essay = raw_essay.split('\n\n')
            assert len(raw_essay) == 2
            assert len(raw_essay[0].split('\n')) == 1
            assert len(raw_essay[1].split('\n')) > 1
            title = raw_essay[0]
            essay = raw_essay[1]
            # change the start and end pos of acs accordingly
            for ac in ordered_ac_info_list:
                ac['start'] = ac['start'] - len(title) - 2
                ac['end'] = ac['end'] - len(title) - 2
                assert essay[ac['start']:ac['end']] == ac['text']
        elif '\n \n' in raw_essay:
            raw_essay = raw_essay.split('\n \n')
            assert len(raw_essay) == 2
            assert len(raw_essay[0].split('\n')) == 1
            assert len(raw_essay[1].split('\n')) > 1
            title = raw_essay[0]
            essay = raw_essay[1]
            # change the start and end pos of acs accordingly
            for ac in ordered_ac_info_list:
                ac['start'] = ac['start'] - len(title) - 3
                ac['end'] = ac['end'] - len(title) - 3
                assert essay[ac['start']:ac['end']] == ac['text']
        elif '\n  \n' in raw_essay:
            raw_essay = raw_essay.split('\n  \n')
            assert len(raw_essay) == 2
            assert len(raw_essay[0].split('\n')) == 1
            assert len(raw_essay[1].split('\n')) > 1
            title = raw_essay[0]
            essay = raw_essay[1]
            # change the start and end pos of acs accordingly
            for ac in ordered_ac_info_list:
                ac['start'] = ac['start'] - len(title) - 4
                ac['end'] = ac['end'] - len(title) - 4
                assert essay[ac['start']:ac['end']] == ac['text']
        else:
            raise ValueError('Strange raw_essay: {}'.format(raw_essay))

        for claim_stance_ann in claim_stance_ann_list:
            assert len(claim_stance_ann.split('\t')) == 2
            claim_stance_ann_id = claim_stance_ann.split('\t')[0]
            claim_stance_info = claim_stance_ann.split('\t')[1]
            assert len(claim_stance_info.split(' ')) == 3
            claim_stance_type = claim_stance_info.split(' ')[2]
            claim_id = claim_stance_info.split(' ')[1]
            assert claim_id.startswith('T')
            assert claim_stance_type in ['Against', 'For']

            for ac in ordered_ac_info_list:
                if ac['ann_id'] == claim_id:
                    ac['type'] = f"{ac['type']}:{claim_stance_type}"
                    break
        
        all_types_set = set([ac['type'] for ac in ordered_ac_info_list])
        for t in all_types_set:
            assert t in ac_type_list

        for ac_info in ordered_ac_info_list:
            assert ac_info['text'] in essay

        essay_ac_info_list.append({
            'idx': essay_id,
            'ac_info_dict_list': ordered_ac_info_list,
            'ar_info_dict_list': ordered_ar_info_list,
            'input_text': essay,
            'topic': title
        })

    return essay_ac_info_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, choices=['essay_level', 'paragraph_level'], default='essay_level', help='essay_level; paragraph_level')
    parser.add_argument('--use_oracle_span', action='store_true', help='use oracle span for the target')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev_ratio', type=float, default=0.1)

    parser.add_argument('--save_dir', type=str, default='input')
    parser.add_argument('--use_mv', action='store_true', help='use multi-view learning')
    args = parser.parse_args()

    args.ar_type_2_prompt = {
        'supports': 'is a "Support" type of argument for',
        'attacks': 'is an "Attack" type of argument for',
    }
    args.ar_prompt_2_type = {
        'is a "Support" type of argument for': 'Support',
        'is an "Attack" type of argument for': 'Attack',
    }

    setup_seed(args.seed)

    args.save_dir = os.path.join(args.save_dir, f'aaec_{args.dataset_type}')
    os.makedirs(args.save_dir, exist_ok=True)

    args.save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')

    if args.use_mv:
        args.save_dir += f'-mv'
    else:
        args.save_dir += f'-sv'

    if args.use_oracle_span:
        args.save_dir += '-os'
    else:
        args.save_dir += '-e2e'

    if args.dataset_type == 'essay_level':
        preprocess_essay_level(args)
    elif args.dataset_type == 'paragraph_level':
        preprocess_paragraph_level(args)


