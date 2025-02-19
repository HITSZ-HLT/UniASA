import argparse
import json
import os
import random
import re
from itertools import chain

from data_utils import produce_source_target
from tqdm import tqdm


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
    # s = re.sub(r'\s{2,}', ' [space] ', s).strip()
    # using regex to remove the space before punctuation
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
            # if ac_text.strip() != ac_info_dict['text'].strip():
            #     print("\nbefore vs after post process")
            #     print(f"[{ac_text}]")
            #     print(f"[{ac_info_dict['text']}]")
            ac_info_dict['start'] = span[0]
            ac_info_dict['end'] = span[1]
            ac_info_dict['text'] = ac_text
    
    return orig_all_data_info_dict_list

def get_info_from_tacl22(args, orig_data_points):
    info_dict_list = []
    for data_point in orig_data_points:
        idx = data_point['id']
        input_text = data_point['input']

        ac_info_dict_list = []
        for node_idx, node in enumerate(data_point['nodes']):
            ac_start = node['anchors'][0]['from']
            ac_end = node['anchors'][0]['to']
            ac_text = input_text[ac_start:ac_end]
            ac_info_dict_list.append({
                'ordered_id': node['id'],
                'type': args.ac_type_2_prompt[node['label'][5:]],
                'start': ac_start,
                'end': ac_end,
                'text': ac_text
            })
            assert node_idx == node['id']
        
        ar_info_dict_list = []
        for edge in data_point['edges']:
            ar_info_dict_list.append({
                'src_ordered_id': edge['target'],
                'tgt_ordered_id': edge['source'],
                'type': args.ar_type_2_prompt[edge['label'][5:]]
            })
        
        info_dict_list.append({
            'idx': idx,
            'ac_info_dict_list': ac_info_dict_list,
            'ar_info_dict_list': ar_info_dict_list,
            'input_text': input_text,
        })
    return info_dict_list

def preprocess_cdcp(args):
    print(f'seed: {args.seed}')
    orig_data_path = 'input/orig_datasets/cdcp-from_tacl22'

    with open(os.path.join(orig_data_path, 'cdcp_train.mrp'), 'r') as f:
        orig_train_data_points = [json.loads(line) for line in f.readlines()]

    with open(os.path.join(orig_data_path, 'cdcp_dev.mrp'), 'r') as f:
        orig_dev_data_points = [json.loads(line) for line in f.readlines()]
    
    with open(os.path.join(orig_data_path, 'cdcp_test.mrp'), 'r') as f:
        orig_test_data_points = [json.loads(line) for line in f.readlines()]

    train_info_dict_list = get_info_from_tacl22(args, orig_train_data_points)
    dev_info_dict_list = get_info_from_tacl22(args, orig_dev_data_points)
    test_info_dict_list = get_info_from_tacl22(args, orig_test_data_points)
    all_data_info_dict_list = train_info_dict_list + dev_info_dict_list + test_info_dict_list
    num_test_data_points = len(test_info_dict_list)
    num_dev_data_points = len(dev_info_dict_list)
    
    def print_stats(data_info_dict_list):
        # print the max number of acs in all essays
        max_ac_num = max([len(_['ac_info_dict_list']) for _ in data_info_dict_list])
        print(f'max_ac_num: {max_ac_num}') # 33
        # print the total number of acs
        total_num_ac = sum([len(_['ac_info_dict_list']) for _ in data_info_dict_list])
        print(f'total_num_ac: {total_num_ac}') # should be 4779
        # print the total number of ars
        total_num_ar = sum([len(_['ar_info_dict_list']) for _ in data_info_dict_list])
        print(f'total_num_ar: {total_num_ar}') # should be 1353
    
    print_stats(all_data_info_dict_list)
    print_stats(train_info_dict_list+dev_info_dict_list)
    # print_stats(dev_info_dict_list)
    print_stats(test_info_dict_list)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    all_data_info_dict_list = post_process(all_data_info_dict_list)

    data_points_list = []
    for info_dict in tqdm(all_data_info_dict_list):
        if args.use_mv == False:
            data_points = produce_source_target(args, info_dict, view_types=['[ASA]'])
        else:
            data_points = produce_source_target(args, info_dict, view_types=['[ACI]', '[ARI]', '[ASA]'])
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
    parser.add_argument('--dataset_type', type=str, choices=['cdcp'], default='cdcp')
    parser.add_argument('--use_oracle_span', action='store_true', help='use oracle span for the target')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev_ratio', type=float, default=0.1)

    parser.add_argument('--save_dir', type=str, default='input')
    parser.add_argument('--use_mv', action='store_true', help='use multi-view learning')
    args = parser.parse_args()

    args.ar_type_2_prompt = {
        'reason': 'is a "Reason" type of argument for',
        'evidence': 'is an "Evidence" type of argument for',
    }
    args.ar_prompt_2_type = {
        'is a "Reason" type of argument for': 'Reason',
        'is an "Evidence" type of argument for': 'Evidence',
    }
    args.ac_type_2_prompt = {
        'value': 'Value',
        'fact': 'Fact',
        'policy': 'Policy',
        'testimony': 'Testimony',
        'reference': 'Reference'
    }

    if args.dataset_type == 'cdcp':
        args.save_dir = os.path.join(args.save_dir, 'cdcp')
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

    preprocess_cdcp(args)
