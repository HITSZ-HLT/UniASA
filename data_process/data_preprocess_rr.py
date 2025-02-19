import argparse
from collections import defaultdict
import json
import os
import random
import re
from itertools import chain
import sys
sys.path.append('./data_process')
from data_utils import produce_source_target_for_rr
from tqdm import tqdm
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('t5-base', local_files_only=True)
except:
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
from transformers.utils import logging
logger = logging.get_logger(__name__)

def read_data(args, data_path, max_sent_len, max_seq_len):
    with open(data_path, 'r') as f:
        raw_data = f.read().strip()
        raw_data_points = raw_data.split('\n\n')
    

    sentences_list = []
    bio_labels_list = []
    sent_pair_ids_list = []
    sent_types_list = []
    sub_ids_list = []
    for raw_data_point in raw_data_points:
        sentences = []
        bio_labels = []
        sent_pair_ids = []
        sent_types = []
        sub_ids = []
        raw_sent_info_list = raw_data_point.split('\n')
        for raw_sent_info in raw_sent_info_list:
            splited_raw_info = raw_sent_info.split('\t')
            sentences.append(splited_raw_info[0])
            bio_labels.append(splited_raw_info[1])
            sent_pair_ids.append(splited_raw_info[2])
            sent_types.append(splited_raw_info[3])
            sub_ids.append(splited_raw_info[4])
        sentences_list.append(sentences)
        bio_labels_list.append(bio_labels)
        sent_pair_ids_list.append(sent_pair_ids)
        sent_types_list.append(sent_types)
        sub_ids_list.append(sub_ids)
        
    len_list =[]
    info_dict_list = []
    for idx, (sentences, bio_labels, sent_pair_ids, sent_types, sub_ids) in tqdm(enumerate(
        zip(
            sentences_list, bio_labels_list, sent_pair_ids_list, sent_types_list, sub_ids_list
            )
        ), total=len(sentences_list)):
        cur_max_sent_len = max_sent_len
        while len(tokenizer.tokenize(' '.join(sentences))) > max_seq_len:
            sentences = [sent[:cur_max_sent_len] for sent in sentences]
            cur_max_sent_len //= 2
        len_list.append(len(tokenizer.tokenize(' '.join(sentences))))

        reply_start_index = sent_types.index('Reply')
        rev_sentences = sentences[:reply_start_index]
        rep_sentences = sentences[reply_start_index:]
        for sent_idx, sent in enumerate(rev_sentences):
            rev_sentences[sent_idx] = f'[# {sent_idx} #] {sent}'
        for sent_idx, sent in enumerate(rep_sentences):
            rep_sentences[sent_idx] = f'[# {sent_idx+len(rev_sentences)} #] {sent}'
        sentences = rev_sentences + rep_sentences
        
        input_text = '[doc_1] ' + ' '.join(rev_sentences) + ' [doc_2] ' + ' '.join(rep_sentences)

        def extract_spans(tags, target_prefix):
            spans = []
            current_span = None

            for i, tag in enumerate(tags):
                if tag.startswith('B-') and tag.split('-')[1] == target_prefix:
                    if current_span:
                        spans.append(current_span)
                    current_span = {"start": i, "end": i + 1}
                elif tag.startswith('I-') and tag.split('-')[1] == target_prefix:
                    if current_span:
                        current_span["end"] = i + 1
                else:
                    if current_span:
                        spans.append(current_span)
                        current_span = None

            if current_span:
                spans.append(current_span)

            return [(span['start'], span['end']) for span in spans]

        review_spans = extract_spans(bio_labels, 'Review')
        reply_spans = extract_spans(bio_labels, 'Reply')
        review_acs_list = [sentences[span[0]:span[1]] for span in review_spans]
        reply_acs_list  = [sentences[span[0]:span[1]] for span in reply_spans]
        review_ac_sent_ids = [' '.join([ac.split(' ')[1] for ac in acs]) for acs in review_acs_list]
        reply_ac_sent_ids = [' '.join([ac.split(' ')[1] for ac in acs]) for acs in reply_acs_list]

        ac_info_dict_list = []
        span_2_ordered_id = {}
        for ac_idx, ac_sent_ids in enumerate(review_ac_sent_ids):
            ac_sent_ids = ac_sent_ids.split(' ')
            span = [int(ac_sent_ids[0]), int(ac_sent_ids[-1]) + 1]
            ac_info_dict_list.append({
                'ordered_id': ac_idx,
                'span': span,
            })
            span_2_ordered_id[tuple(span)] = ac_idx
        for ac_idx, ac_sent_ids in enumerate(reply_ac_sent_ids):
            ac_sent_ids = ac_sent_ids.split(' ')
            span = [int(ac_sent_ids[0]), int(ac_sent_ids[-1]) + 1]
            ac_info_dict_list.append({
                'ordered_id': ac_idx + len(review_ac_sent_ids),
                'span': span,
            })
            span_2_ordered_id[tuple(span)] = ac_idx + len(review_ac_sent_ids)
        assert len(ac_info_dict_list) == len(span_2_ordered_id)

        rev_sent_pair_ids = sent_pair_ids[:reply_start_index]
        rep_sent_pair_ids = sent_pair_ids[reply_start_index:]

        def extract_spans_pairs(review_label_list, rebuttal_label_list):
            pairs = []

            for i in range(len(review_label_list)):
                if review_label_list[i].startswith('B-'):
                    start_review = i
                    end_review = i
                    while end_review < len(review_label_list) - 1 and review_label_list[end_review + 1].startswith('I-'):
                        end_review += 1
                    for j in range(len(rebuttal_label_list)):
                        if rebuttal_label_list[j].startswith('B-'):
                            start_rebuttal = j
                            end_rebuttal = j
                            while end_rebuttal < len(rebuttal_label_list) - 1 and rebuttal_label_list[end_rebuttal + 1].startswith('I-'):
                                end_rebuttal += 1
                            if review_label_list[i] == rebuttal_label_list[j]:
                                pairs.append(((start_review, end_review + 1), (start_rebuttal, end_rebuttal + 1)))
            return pairs

        pairs = extract_spans_pairs(rev_sent_pair_ids, rep_sent_pair_ids)
        pairs = [[p[0], (p[1][0]+reply_start_index, p[1][1]+reply_start_index)] for p in pairs]
        # review_spans_set = set(review_spans)
        # reply_spans_set = set(reply_spans)
        # minus reply_start_index because the reply_spans are started from review.
        # check
        for pair in pairs:
            assert pair[0] in span_2_ordered_id
            assert pair[1] in span_2_ordered_id
        
        ar_info_dict_list = []
        for pair in pairs:
            ar_info_dict_list.append({
                'src_ordered_id': span_2_ordered_id[pair[0]],
                'tgt_ordered_id': span_2_ordered_id[pair[1]],
                'type': args.ar_type_2_prompt['Argument-Pair']
            })

        info_dict_list.append({
            'idx': idx,
            'ac_info_dict_list': ac_info_dict_list,
            'ar_info_dict_list': ar_info_dict_list,
            'input_text': input_text,
        })
    
    print()
    return info_dict_list

def preprocess_rr(args):
    print(f'seed: {args.seed}')
    orig_data_path = 'input/orig_datasets/RR-submission-v2'

    print('Reading train data ...')
    train_info_dict_list = read_data(args, os.path.join(orig_data_path, 'train.txt'), args.max_sent_len, args.max_seq_len)
    print('Reading validation data ...')
    validation_info_dict_list = read_data(args, os.path.join(orig_data_path, 'dev.txt'), args.max_sent_len, args.max_seq_len)
    print('Reading test data ...')
    test_info_dict_list = read_data(args, os.path.join(orig_data_path, 'test.txt'), args.max_sent_len, args.max_seq_len)

    all_data_info_dict_list = train_info_dict_list + validation_info_dict_list + test_info_dict_list
    num_test_data_points = len(test_info_dict_list)
    num_dev_data_points = len(validation_info_dict_list)

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
    print_stats(train_info_dict_list)
    print_stats(validation_info_dict_list)
    print_stats(test_info_dict_list)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # all_data_info_dict_list = post_process(all_data_info_dict_list)

    data_points_list = []
    for info_dict in tqdm(all_data_info_dict_list):
        if args.use_mv == False:
            data_points = produce_source_target_for_rr(args, info_dict, view_types=['[ASA]'])
        else:
            data_points = produce_source_target_for_rr(args, info_dict, view_types=['[ACI]', '[ARI]', '[ASA]'])
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
    parser.add_argument('--dataset_type', type=str, choices=['rr'], default='rr')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--save_dir', type=str, default='input')
    parser.add_argument('--use_mv', action='store_true', help='use multi-view learning')
    parser.add_argument('--max_sent_len', type=int, default=200)
    parser.add_argument('--max_seq_len', type=int, default=2500)

    args = parser.parse_args()

    args.ar_type_2_prompt = {
      'Argument-Pair': 'can form an "Argument-Pair" with'
    }
    args.ar_prompt_2_type = {v: k for k, v in args.ar_type_2_prompt.items()}

    # setup_seed(args.seed)

    # 3407 1996 43 1024 2049
    # python data_process/data_preprocess_cdcp.py --dataset_type cdcp --view_type mv
    # python -m debugpy --listen 5432 --wait-for-client data_process/data_preprocess_cdcp.py --dataset_type cdcp --view_type mv
    
    if args.dataset_type == 'rr':
        args.save_dir = os.path.join(args.save_dir, 'rr')
    os.makedirs(args.save_dir, exist_ok=True)
    if args.use_mv:
        args.save_dir = os.path.join(args.save_dir, 'mv')
    else:
        args.save_dir = os.path.join(args.save_dir, 'sv')

    if args.max_sent_len:
        args.save_dir = args.save_dir + f'-len_{args.max_sent_len}'
    
    if args.max_seq_len:
        args.save_dir = args.save_dir + f'-seq_{args.max_seq_len}'

    preprocess_rr(args)
