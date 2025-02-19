import os
from copy import deepcopy
import random
import json
import nltk
from nltk.corpus import stopwords
import numpy as np
import warnings
from collections import defaultdict
from transformers import AutoTokenizer

e2e_task_description_prompt = '''[ASA] Please do an argumentation structure analysis task.'''
aci_task_description_prompt = '''[ACI] Please do an argument component identification task.'''
ari_task_description_prompt = '''[ARI] Please do an argumentative relation extraction task.'''

task_prompts = {
    '[ASA]': e2e_task_description_prompt,
    '[ACI]': aci_task_description_prompt,
    '[ARI]': ari_task_description_prompt
}

try:
    tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-base', local_files_only=True)
except:
    tokenizer = AutoTokenizer.from_pretrained('google/long-t5-tglobal-base')



def find_unk_replacements(orig_str, decoded_str):
    i, j = 0, 0
    replacements = {}

    while i < len(orig_str) and j < len(decoded_str):
        if j < len(decoded_str) - 4 and decoded_str[j:j+5] == '<unk>':
            unk_pos_span = (j, j+5)
            start = i
            j += 5 
            while i < len(orig_str) and orig_str[i] != decoded_str[j]:
                i += 1
            replacements[str(unk_pos_span)] = orig_str[start:i]
        elif orig_str[i] == decoded_str[j]:
            i += 1
            j += 1
        else:
            print("------------------")
            print("Unexpected discrepancy between strings")
            print("orig_str:")
            print(orig_str)
            print("decoded_str:")
            print(decoded_str)
            break

    return replacements


def produce_target(args, view_type, ac_info_dict_list, ar_info_dict_list):
    src_ac_id2related_ars = defaultdict(dict)
    for ar in ar_info_dict_list:
        if ar['src_ordered_id'] not in src_ac_id2related_ars:
            src_ac_id2related_ars[ar['src_ordered_id']] = []
        src_ac_id2related_ars[ar['src_ordered_id']].append(ar)

    tgt_ac_id2related_ars = defaultdict(dict)
    for ar in ar_info_dict_list:
        if ar['tgt_ordered_id'] not in tgt_ac_id2related_ars:
            tgt_ac_id2related_ars[ar['tgt_ordered_id']] = []
            tgt_ac_id2related_ars[ar['tgt_ordered_id']] = []
        tgt_ac_id2related_ars[ar['tgt_ordered_id']].append(ar)

    target = f'{view_type} '
    # view_types = ['[ACI]', '[ARI]', '[ASA]']
    for ac in ac_info_dict_list:
        sub_target = ''
        if args.dataset_type == 'qam':
            ac_item = ac["ordered_id"]
            ac_type = ac["type"]
        elif args.dataset_type == 'rr':
            ac_item = ' '.join(map(str, range(ac['span'][0], ac['span'][1])))
            ac_type = "Argument"
        else:
            if args.use_oracle_span == True:
                ac_item = ac["ordered_id"]
            else:
                ac_item = ac["text"]
            ac_type = ac["type"]
        if view_type == '[ACI]':
            sub_target += f'The type of [# {ac_item} #] is "{ac_type}".'
        elif view_type == '[ARI]':
            sub_target += f'[# {ac_item} #]'
            if ac["ordered_id"] in src_ac_id2related_ars:
                is_first = True
                for ar in src_ac_id2related_ars[ac["ordered_id"]]:
                    if args.dataset_type == 'qam':
                        tgt_ac_item = ar["tgt_ordered_id"]
                    elif args.dataset_type == 'rr':
                        tgt_span = ac_info_dict_list[ar['tgt_ordered_id']]['span']
                        tgt_ac_item = ' '.join(map(str, range(tgt_span[0], tgt_span[1])))
                    else:
                        if args.use_oracle_span == True:
                            tgt_ac_item = ar["tgt_ordered_id"]
                        else:
                            tgt_ac_item = ac_info_dict_list[ar["tgt_ordered_id"]]['text']
                    if is_first:
                        sub_target += f' '
                        is_first = False
                    else:
                        sub_target += f' and '
                    sub_target += f'{ar["type"]} [# {tgt_ac_item} #],'
            else:
                sub_target += ' produces no relation.'
            if sub_target.endswith(','):
                sub_target = sub_target[:-1]
                sub_target += '.'
        elif view_type == '[ASA]':
            sub_target += f'The type of [# {ac_item} #] is "{ac_type}". It'
            if ac["ordered_id"] in src_ac_id2related_ars:
                is_first = True
                for ar in src_ac_id2related_ars[ac["ordered_id"]]:
                    if args.dataset_type == 'qam':
                        tgt_ac_item = ar["tgt_ordered_id"]
                    elif args.dataset_type == 'rr':
                        tgt_span = ac_info_dict_list[ar['tgt_ordered_id']]['span']
                        tgt_ac_item = ' '.join(map(str, range(tgt_span[0], tgt_span[1])))
                    else:
                        tgt_type = ac_info_dict_list[ar["tgt_ordered_id"]]['type']
                        if args.use_oracle_span == True:
                            tgt_ac_item = ar["tgt_ordered_id"]
                        else:
                            tgt_ac_item = ac_info_dict_list[ar["tgt_ordered_id"]]['text']
                    if is_first:
                        sub_target += f' '
                        is_first = False
                    else:
                        sub_target += f' and '
                    if args.dataset_type == 'qam' or args.dataset_type == 'rr':    
                        sub_target += f'{ar["type"]} [# {tgt_ac_item} #],'
                    else:
                        sub_target += f'{ar["type"]} [# {tgt_ac_item} #] ("{tgt_type}"),'
            else:
                sub_target += ' produces no relation.'
            if sub_target.endswith(','):
                sub_target = sub_target[:-1]
                sub_target += '.'

        if sub_target:
            target += f'{sub_target} [SEP] '
    
    target = target.strip()

    if target == view_type:
        target += ' None'

    return target


def produce_source_target(args, data_points_info_dict, view_types=['[ACI]', '[ARI]', '[ASA]']):
    ac_info_dict_list = data_points_info_dict['ac_info_dict_list']
    ar_info_dict_list = data_points_info_dict['ar_info_dict_list']
    instance_idx = data_points_info_dict['idx']
    input_text = data_points_info_dict['input_text']
    given_topic = data_points_info_dict.get('topic', None)
    orig_input_text = input_text

    data_points = []

    for view_type in view_types:
        if view_type not in task_prompts:
            raise ValueError(f'Unknown view_type: {view_type}')

        source = task_prompts[view_type] + ' '

        if args.use_oracle_span == True:
            aci_labels = [[d["ordered_id"], d["type"]] for d in ac_info_dict_list]
            ari_labels = [
                [
                    d["src_ordered_id"],
                    args.ar_prompt_2_type[d["type"]], 
                    d["tgt_ordered_id"]
                ] 
                for d in ar_info_dict_list
            ]
            truth_arict_list = [
                [
                    d["src_ordered_id"], 
                    ac_info_dict_list[d["src_ordered_id"]]['type'], 
                    args.ar_prompt_2_type[d["type"]], 
                    d["tgt_ordered_id"], 
                    ac_info_dict_list[d["tgt_ordered_id"]]['type']
                ] 
                for d in ar_info_dict_list
            ]
        else:
            aci_labels = [[d["ordered_id"], d["text"], [d["start"], d["end"]], d["type"]] for d in ac_info_dict_list]
            for i, d in enumerate(ac_info_dict_list):
                assert orig_input_text[d["start"]:d["end"]] == d["text"]
                assert i == d["ordered_id"]
            ari_labels = [
                [
                    d["src_ordered_id"],
                    [ac_info_dict_list[d["src_ordered_id"]]["start"], ac_info_dict_list[d["src_ordered_id"]]["end"]],
                    args.ar_prompt_2_type[d["type"]], 
                    d["tgt_ordered_id"],
                    [ac_info_dict_list[d["tgt_ordered_id"]]["start"], ac_info_dict_list[d["tgt_ordered_id"]]["end"]]
                ] 
                for d in ar_info_dict_list
            ]
            truth_arict_list = [
                [
                    d["src_ordered_id"], 
                    [ac_info_dict_list[d["src_ordered_id"]]["start"], ac_info_dict_list[d["src_ordered_id"]]["end"]],
                    ac_info_dict_list[d["src_ordered_id"]]['type'], 
                    args.ar_prompt_2_type[d["type"]], 
                    d["tgt_ordered_id"], 
                    [ac_info_dict_list[d["tgt_ordered_id"]]["start"], ac_info_dict_list[d["tgt_ordered_id"]]["end"]],
                    ac_info_dict_list[d["tgt_ordered_id"]]['type']
                ] 
                for d in ar_info_dict_list
            ]

        # specify the position of each ac in the input text
        cur_input_text = orig_input_text
        if args.use_oracle_span == True:
            start_pos, end_pos = 0, 0
            for ac_idx, ac_info_dict in enumerate(ac_info_dict_list):
                start_pos = cur_input_text.find(ac_info_dict['text'], end_pos) # two identical acs may exist in the essay, how many? only three essays 186 264 286
                assert start_pos != -1
                end_pos = start_pos + len(ac_info_dict['text'])
                # add an id token [i] to the beginning of each ac, also insert a [AC] token before each ac, and a [/AC] token after each ac.
                ac_formatted = f'[AC] [# {ac_info_dict["ordered_id"]} #] {ac_info_dict["text"]} [/AC]'
                # replace the ac with the ac_formated
                cur_input_text = cur_input_text[:start_pos] + ac_formatted + cur_input_text[end_pos:]
                end_pos = start_pos + len(ac_formatted)
        else:
            pass
        cur_input_text = cur_input_text.replace('\n', ' [NEW_LINE] ') # Important
        source += '| Topic: ' + (given_topic + ' | ' if given_topic else 'None | ')
        source += 'Text: ' + cur_input_text
        source = source.strip()

        target = produce_target(args, view_type, ac_info_dict_list, ar_info_dict_list)

        # assert '  ' not in source
        # assert '  ' not in target

        aci_labels = sorted(aci_labels)
        ari_labels = sorted(ari_labels)
        truth_arict_list = sorted(truth_arict_list)

        decoded_input_text = tokenizer.decode(tokenizer(orig_input_text)['input_ids'], clean_up_tokenization_spaces=False)
        # remove </s> token at the end
        decoded_input_text = decoded_input_text[:-4]
        if '<unk>' in decoded_input_text:
            unk_replacements = find_unk_replacements(orig_input_text, decoded_input_text)
        else:
            unk_replacements = {}

        data_points.append({
            'source': source,
            'target': target,
            'idx': instance_idx,
            'aci_labels': json.dumps(aci_labels),
            'ari_labels': json.dumps(ari_labels),
            'arict_labels': json.dumps(truth_arict_list),
            'view_type': view_type,
            'orig_input_text': orig_input_text,
            'decoded_input_text': decoded_input_text,
            'unk_replacements': json.dumps(unk_replacements),
            'given_topic': given_topic
        })

    return data_points

def produce_source_target_for_qam(args, data_points_info_dict, view_types=['[ACI]', '[ARI]', '[ASA]']):
    ac_info_dict_list = data_points_info_dict['ac_info_dict_list']
    ar_info_dict_list = data_points_info_dict['ar_info_dict_list']
    instance_idx = data_points_info_dict['idx']
    input_text = data_points_info_dict['input_text']
    given_topic = data_points_info_dict.get('topic', None)
    orig_input_text = input_text

    data_points = []

    for view_type in view_types:
        if view_type not in task_prompts:
            raise ValueError(f'Unknown view_type: {view_type}')

        source = task_prompts[view_type] + ' '
        
        aci_labels = [[d["ordered_id"], d["type"]] for d in ac_info_dict_list]
        truth_aci_dict = {d["ordered_id"]: d["type"] for d in ac_info_dict_list}
        ari_labels = [
            [
                d["src_ordered_id"], 
                truth_aci_dict[d["src_ordered_id"]], 
                args.ar_prompt_2_type[d["type"]], 
                d["tgt_ordered_id"],
                "dummy"
            ] 
            for d in ar_info_dict_list
        ]

        # specify the position of each ac in the input text
        cur_input_text = ''
        sents = orig_input_text.split(' [sent] ')
        for sent_idx, sent in enumerate(sents):
            cur_input_text += f'[AC] [# {sent_idx} #] {sent} [/AC] '

        cur_input_text = cur_input_text.strip()
        cur_input_text = cur_input_text.replace('\n', ' [NEW_LINE] ') # Important
        source += '| Topic: ' + (given_topic + ' | ' if given_topic else 'None | ')
        source += 'Text: ' + cur_input_text
        source = source.strip()

        target = produce_target(args, view_type, ac_info_dict_list, ar_info_dict_list)

        # assert '  ' not in source
        # assert '  ' not in target

        aci_labels = sorted(aci_labels)
        ari_labels = sorted(ari_labels)

        unk_replacements = {}
        decoded_input_text = orig_input_text

        data_points.append({
            'source': source,
            'target': target,
            'idx': instance_idx,
            'aci_labels': json.dumps(aci_labels),
            'ari_labels': json.dumps(ari_labels),
            'view_type': view_type,
            'orig_input_text': orig_input_text,
            'decoded_input_text': decoded_input_text,
            'unk_replacements': json.dumps(unk_replacements),
            'given_topic': given_topic
        })

    return data_points



def produce_source_target_for_rr(args, data_points_info_dict, view_types=['[ACI]', '[ARI]', '[ASA]']):
    ac_info_dict_list = data_points_info_dict['ac_info_dict_list']
    ar_info_dict_list = data_points_info_dict['ar_info_dict_list']
    instance_idx = data_points_info_dict['idx']
    input_text = data_points_info_dict['input_text']
    given_topic = data_points_info_dict.get('topic', None)
    orig_input_text = input_text

    data_points = []

    for view_type in view_types:
        if view_type not in task_prompts:
            raise ValueError(f'Unknown view_type: {view_type}')

        source = task_prompts[view_type] + ' '

        aci_labels = [d["span"] for d in ac_info_dict_list]

        ari_labels = [
            [
                ac_info_dict_list[d["src_ordered_id"]]['span'], 
                ac_info_dict_list[d["tgt_ordered_id"]]['span'],
            ] 
            for d in ar_info_dict_list
        ]

        # specify the position of each ac in the input text
        cur_input_text = input_text

        cur_input_text = cur_input_text.replace('\n', ' [NEW_LINE] ') # Important
        source += '| Topic: ' + (given_topic + ' | ' if given_topic else 'None | ')
        source += 'Text: ' + cur_input_text
        source = source.strip()

        target = produce_target(args, view_type, ac_info_dict_list, ar_info_dict_list)

        # assert '  ' not in source
        # assert '  ' not in target

        aci_labels = sorted(aci_labels)
        ari_labels = sorted(ari_labels)

        unk_replacements = {}
        decoded_input_text = orig_input_text

        data_points.append({
            'source': source,
            'target': target,
            'idx': instance_idx,
            'aci_labels': json.dumps(aci_labels),
            'ari_labels': json.dumps(ari_labels),
            'view_type': view_type,
            'orig_input_text': orig_input_text,
            'decoded_input_text': decoded_input_text,
            'unk_replacements': json.dumps(unk_replacements),
            'given_topic': given_topic
        })

    return data_points
