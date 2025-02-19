import argparse
import json
import os
import random
import re
from itertools import chain

from data_utils import produce_source_target
from tqdm import tqdm
import xml.etree.ElementTree as ET


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
                # 'type': node['label'][4:],
                'type': args.ac_type_2_prompt[node['label'][4:]],
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
                # 'type': edge['label'][4:],
                'type': args.ar_type_2_prompt[edge['label'][4:]],
            })
        
        info_dict_list.append({
            'idx': idx,
            'ac_info_dict_list': ac_info_dict_list,
            'ar_info_dict_list': ar_info_dict_list,
            'input_text': input_text,
        })
    return info_dict_list

def get_topic_descriptions(text):
    topic_descriptions = {}
    s = text.find("## List of the topics")
    text = text[s+len("## List of the topics"):]

    items = text.split('*')
    items = [item.strip() for item in items if item.strip() != '']

    pattern = r"`(.*?)`\s*de:.*?\sen: (.*?)\s*$"
    for text in items:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            topic_id = match.group(1)
            topic_description = match.group(2)
            topic_descriptions[topic_id] = topic_description.strip()
    
    return topic_descriptions

def preprocess_mtc(args):
    print(f'seed: {args.seed}')
    orig_data_path = 'input/orig_datasets/mtc-from_tacl22'

    with open(os.path.join(orig_data_path, f'mtc.cv{args.fold}.train.mrp'), 'r') as f:
        orig_train_data_points = [json.loads(line) for line in f.readlines()]

    with open(os.path.join(orig_data_path, f'mtc.cv{args.fold}.dev.mrp'), 'r') as f:
        orig_dev_data_points = [json.loads(line) for line in f.readlines()]
    
    with open(os.path.join(orig_data_path, f'mtc.cv{args.fold}.test.mrp'), 'r') as f:
        orig_test_data_points = [json.loads(line) for line in f.readlines()]

    train_info_dict_list = get_info_from_tacl22(args, orig_train_data_points)
    dev_info_dict_list = get_info_from_tacl22(args, orig_dev_data_points)
    test_info_dict_list = get_info_from_tacl22(args, orig_test_data_points)
    all_data_info_dict_list = train_info_dict_list + dev_info_dict_list + test_info_dict_list

    id_to_topic_id = {}
    raw_data_dir = 'input/orig_datasets/arg-microtexts-master/corpus/en'
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".xml"):
            filepath = os.path.join(raw_data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = file.read()
                root = ET.fromstring(data)
                arggraph_id = root.get('id')
                topic_id = root.get('topic_id')
                id_to_topic_id[arggraph_id] = topic_id
    
    for _ in all_data_info_dict_list:
        if id_to_topic_id[_['idx']]:
            _['topic'] = ' '.join(id_to_topic_id[_['idx']].split('_'))
        else:
            _['topic'] = 'None'


    num_test_data_points = len(test_info_dict_list)
    num_dev_data_points = len(dev_info_dict_list)
    
    # print the max number of acs in all essays
    max_ac_num = max([len(_['ac_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'max_ac_num: {max_ac_num}') # 19
    # print the total number of acs
    total_num_ac = sum([len(_['ac_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'total_num_ac: {total_num_ac}') # should be 3279
    # print the total number of ars
    total_num_ar = sum([len(_['ar_info_dict_list']) for _ in all_data_info_dict_list])
    print(f'total_num_ar: {total_num_ar}') # should be 2060

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
    parser.add_argument('--dataset_type', type=str, choices=['mtc'], default='mtc')
    parser.add_argument('--use_oracle_span', action='store_true', help='use oracle span for the target')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev_ratio', type=float, default=0.1)

    parser.add_argument('--save_dir', type=str, default='input')
    parser.add_argument('--use_mv', action='store_true', help='use multi-view learning')
    args = parser.parse_args()

    args.ar_type_2_prompt = {
        'exa': 'is an "Example" type of argument for',
        'sup': 'is a "Support" type of argument for',
        'und': 'is an "Undercut" type of argument for',
        'reb': 'is a "Rebut" type of argument for',
    }
    args.ar_prompt_2_type = {
        'is an "Example" type of argument for': 'Example',
        'is a "Support" type of argument for': 'Support',
        'is an "Undercut" type of argument for': 'Undercut',
        'is a "Rebut" type of argument for': 'Rebut',
    }
    args.ac_type_2_prompt = {
        'opp': 'Opponent',
        'pro': 'Proponent',
    }
    args.save_dir = os.path.join(args.save_dir, args.dataset_type)
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = os.path.join(args.save_dir, f'seed_{args.seed}')
    args.save_dir += f'-fold_{args.fold}'

    if args.use_mv:
        args.save_dir += f'-mv'
    else:
        args.save_dir += f'-sv'

    if args.use_oracle_span:
        args.save_dir += '-os'
    else:
        args.save_dir += '-e2e'

    preprocess_mtc(args)
