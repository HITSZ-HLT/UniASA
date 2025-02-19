import ast
import numpy as np
import nltk
import re
import json
from collections import defaultdict, Counter
from transformers.utils import logging
import fuzzysearch
import os
logger = logging.get_logger(__name__)


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

def get_spans_fuzzy(orig_essay, ac_text_list):
    spans = []
    start, end = 0, 0
    valid_ac_flags = []
    for ac_text in ac_text_list:
        # start = orig_essay.find(ac_text, end)
        matches = fuzzysearch.find_near_matches(ac_text, orig_essay[end:], max_l_dist=int(len(ac_text)*0.1))

        if matches:
            best_match = min(matches, key=lambda match: match.dist)
            start = end + best_match.start
            end = end + best_match.end
            spans.append([start, end])
            valid_ac_flags.append(True)
        else:
            valid_ac_flags.append(False)
    # valid_ac_flags means whether the ac_text is valid in orig_essay
    return spans, valid_ac_flags

class AsaEvaluation:
    def __init__(self, data_args, eval_set, decoded_preds, am_f1_metric):
        self.data_args = data_args
        self.eval_set = eval_set
        self.decoded_preds = decoded_preds
        self.am_f1_metric = am_f1_metric

    def get_true_labels(self):
        '''
        Only use e2e forward labels for evaluation
        '''
        aci_labels = self.eval_set['aci_labels']
        ari_labels = self.eval_set['ari_labels']
        view_type_list = self.eval_set['view_type']

        reduced_aci_labels = []
        reduced_ari_labels = []
        for view_type, truth_aci, truth_ari in zip(view_type_list, aci_labels, ari_labels):
            truth_aci = json.loads(truth_aci)
            truth_ari = json.loads(truth_ari)
            if view_type.startswith('[ASA]'):
                if self.data_args.task_type == 'AM':
                    if self.data_args.use_oracle_span:
                        reduced_aci_labels.append(truth_aci)
                        reduced_ari_labels.append(truth_ari)
                    else:
                        reduced_aci_labels.append([_[2:] for _ in truth_aci])
                        reduced_ari_labels.append([[_[1], _[2], _[4]] for _ in truth_ari])
                elif self.data_args.task_type == 'APE':
                    reduced_aci_labels.append(truth_aci)
                    reduced_ari_labels.append(truth_ari)
                elif self.data_args.task_type == 'AQE':
                    reduced_aci_labels.append(truth_aci)
                    reduced_ari_labels.append(truth_ari)
                else:
                    raise ValueError("Invalid task type.")
        
        if self.data_args.task_type == 'AM':
            arict_labels = self.eval_set['arict_labels'] 
            assert len(arict_labels) == len(ari_labels)
            reduced_arict_labels = []
            for view_type, truth_arict in zip(view_type_list, arict_labels):
                truth_arict = json.loads(truth_arict)
                if view_type.startswith('[ASA]'):
                    if self.data_args.use_oracle_span:
                        reduced_arict_labels.append(truth_arict)
                    else:
                        reduced_arict_labels.append([[_[1], _[2], _[3], _[5], _[6]] for _ in truth_arict])
        else:
            reduced_arict_labels = None

        return reduced_aci_labels, reduced_ari_labels, reduced_arict_labels

    def get_pred_labels(self, pred_aci_list, pred_ari_triple_list, sample_idx, voting_threshold):
        assert len(pred_aci_list) == len(pred_ari_triple_list) == len(sample_idx)

        grouped_pred_aci, grouped_pred_ari = defaultdict(Counter), defaultdict(Counter)

        for idx, pred_aci, pred_ari_to in zip(sample_idx, pred_aci_list, pred_ari_triple_list):
            grouped_pred_aci[idx].update([tuple(_) if isinstance(_, list) else _ for _ in pred_aci])
            grouped_pred_ari[idx].update([tuple([tuple(__) if isinstance(__, list) else __ for __ in _]) for _ in pred_ari_to])

        filtered_grouped_pred_aci = {
            sample_idx: {k: v for k, v in pred_aci_counter.items() if v >= voting_threshold}
            for sample_idx, pred_aci_counter in grouped_pred_aci.items()
        }

        filtered_grouped_pred_ari = {
            sample_idx: {k: v for k, v in pred_ari_counter.items() if v >= voting_threshold}
            for sample_idx, pred_ari_counter in grouped_pred_ari.items()
        }

        reduced_pred_aci_list = [[list(item) for item in group.keys()] for group in filtered_grouped_pred_aci.values()]
        reduced_pred_ari_triple_list = [[list(item) for item in group.keys()] for group in filtered_grouped_pred_ari.values()]

        # span_set_list = [set([_[0] for _ in sublist]) for sublist in reduced_pred_aci_list]
        # reduced_pred_ari_triple_list = [
        #     [item for item in sublist if item[0] in span_set_list[i] and item[2] in span_set_list[i]]
        #     for i, sublist in enumerate(reduced_pred_ari_triple_list)
        # ]

        return reduced_pred_aci_list, reduced_pred_ari_triple_list
    

    def get_labels_and_preds(self):
        sample_idx = self.eval_set['idx']

        aci_labels_list, ari_labels_list, arict_labels_list = self.get_true_labels()

        if self.data_args.task_type == 'AM':
            ac_type_set = set([_[-1] for sublist in aci_labels_list for _ in sublist])
            ar_type_set = set([_[1] for sublist in ari_labels_list for _ in sublist])
        elif self.data_args.task_type == 'APE':
            ac_type_set, ar_type_set = None, None
        elif self.data_args.task_type == 'AQE':
            ac_type_set = set([_[-1] for sublist in aci_labels_list for _ in sublist])
            ar_type_set = set([_[2] for sublist in ari_labels_list for _ in sublist])

        aci_preds_list = []
        ari_preds_list = []
        for (
            source,
            pred_target,
            view_type,
            orig_input_text,
            decoded_input_text,
            unk_replacements
        ) in zip(
                self.eval_set['source'],
                self.decoded_preds,
                self.eval_set['view_type'],
                self.eval_set['orig_input_text'],
                self.eval_set['decoded_input_text'],
                self.eval_set['unk_replacements']
            ):
            aci_preds, ari_preds = self.parse_target_seq(
                source=source, 
                target=pred_target,
                ac_type_set=ac_type_set,
                ar_type_set=ar_type_set,
                view_type=view_type,
                orig_input_text=orig_input_text,
                decoded_input_text=decoded_input_text,
                unk_replacements=unk_replacements,
                task_type=self.data_args.task_type,
                use_oracle_span=self.data_args.use_oracle_span,
                use_fuzzy_search=self.data_args.use_fuzzy_search
            )
            aci_preds_list.append(aci_preds)
            ari_preds_list.append(ari_preds)

        aci_preds_list, ari_preds_list = self.get_pred_labels(
            aci_preds_list, ari_preds_list, sample_idx, self.data_args.voting_threshold
        )
        return aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list, arict_labels_list


    def compute_macro_f1(self, preds_list, labels_list, eval_mode):
        if eval_mode == 'aci':
            type_label_position = 1
        elif eval_mode == 'ari':
            type_label_position = 1
        else:
            raise ValueError("Invalid eval_mode.")

        # calculate F1 score for each category
        all_types = list(set(label_tuple[type_label_position] 
                                for labels in labels_list 
                                for label_tuple in labels))

        # split according to categories
        preds_list_by_type, labels_list_by_type = [], []
        for preds, labels in zip(preds_list, labels_list):
            preds_by_type, labels_by_type = defaultdict(list), defaultdict(list)
            for item in preds:
                preds_by_type[item[type_label_position]].append(item)
            for item in labels:
                labels_by_type[item[type_label_position]].append(item)
            # add empty list for types that are not in the sample
            for t in all_types:
                preds_by_type[t], labels_by_type[t]
            preds_list_by_type.append(preds_by_type)
            labels_list_by_type.append(labels_by_type)
        
        all_f1_res = {}
        for t in all_types:
            preds_of_t_list = [preds_by_type[t] for preds_by_type in preds_list_by_type]
            labels_of_t_list = [labels_by_type[t] for labels_by_type in labels_list_by_type]
            f1_res = self.am_f1_metric.compute(predictions=preds_of_t_list, references=labels_of_t_list)
            all_f1_res[t] = f1_res

        all_f1_res['f1'] = np.mean([f1_res['f1'] for f1_res in all_f1_res.values()]).tolist()
        return all_f1_res

    def add_key_prefix(self, prefix, res_dict):
        return {f"{prefix}_{k}": v for k, v in res_dict.items()}

    def cal_score(self):
        aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list, arict_labels_list = \
            self.get_labels_and_preds()

        # create a mapping dictionary from index to type for each sublist
        aci_preds_dict_list = [dict(sublist) for sublist in aci_preds_list]
        if self.data_args.task_type == "AQE":
            arict_preds_list = [
                # [
                #     [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                #     if item[0] in aci_preds_dict_list[i]
                #     else []
                #     for item in sublist
                # ]
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                    for item in sublist if item[0] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(ari_preds_list)
            ]
        elif self.data_args.task_type == "AM":
            arict_preds_list = [
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], aci_preds_dict_list[i][item[2]]]
                    for item in sublist if item[0] in aci_preds_dict_list[i] and item[2] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(ari_preds_list)
            ]
        else:
            arict_preds_list = []

        # change tuple into list
        aci_preds_list = self.convert_tuples_to_lists(aci_preds_list)
        ari_preds_list = self.convert_tuples_to_lists(ari_preds_list)

        all_eval_res = {}

        aci_micro_f1_res = self.am_f1_metric.compute(predictions=aci_preds_list, references=aci_labels_list)
        if self.data_args.task_type == 'AQE':
            ari_micro_f1_res = self.am_f1_metric.compute(predictions=arict_preds_list, references=ari_labels_list)
        else:
            ari_micro_f1_res = self.am_f1_metric.compute(predictions=ari_preds_list, references=ari_labels_list)

        if arict_labels_list:
            arict_preds_list = self.convert_tuples_to_lists(arict_preds_list)
            arict_f1_res = self.am_f1_metric.compute(predictions=arict_preds_list, references=arict_labels_list)
        else:
            arict_f1_res = {}
        # aci_micro_f1_F, aci_micro_f1_P, aci_micro_f1_R
        all_eval_res.update(self.add_key_prefix('aci_micro', aci_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('ari_micro', ari_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('arict', arict_f1_res))
        
        # if self.data_args.task_type != 'AM':
        #     return all_eval_res
        if self.data_args.task_type == 'AQE':
            all_eval_res['aqe_f1'] = all_eval_res['ari_micro_f1']
            all_eval_res['aqe_precision'] = all_eval_res['ari_micro_precision']
            all_eval_res['aqe_recall'] = all_eval_res['ari_micro_recall']
        elif self.data_args.task_type == 'APE':
            all_eval_res['ape_f1'] = all_eval_res['ari_micro_f1']
            all_eval_res['ape_precision'] = all_eval_res['ari_micro_precision']
            all_eval_res['ape_recall'] = all_eval_res['ari_micro_recall']
        elif self.data_args.task_type == 'AM':
            all_eval_res.update(self.cal_macro_f1_score(aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list))
            if not self.data_args.use_oracle_span:
                all_eval_res.update(self.cal_acl17_f1_score(aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list))
            else:
                file_path = self.data_args.test_file if self.data_args.test_file else self.data_args.train_file
                if 'aaec_paragraph_level' in file_path:
                    all_eval_res.update(self.cal_acl19_f1_score(aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list))
        return all_eval_res
    

    def cal_macro_f1_score(self, aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list):
        all_eval_res = {}
        if not self.data_args.use_oracle_span:
            span_preds_list = [[[item[0]] for item in sublist] for sublist in aci_preds_list]
            span_labels_list = [[[item[0]] for item in sublist] for sublist in aci_labels_list]
            span_f1_res = self.am_f1_metric.compute(predictions=span_preds_list, references=span_labels_list)
            all_eval_res.update(self.add_key_prefix('span', span_f1_res))

        link_preds_list = [[[item[0], item[2]] for item in sublist] for sublist in ari_preds_list]
        link_labels_list = [[[item[0], item[2]] for item in sublist] for sublist in ari_labels_list]
        link_f1_res = self.am_f1_metric.compute(predictions=link_preds_list, references=link_labels_list)
        all_eval_res.update(self.add_key_prefix('link', link_f1_res))

        def post_process_macro_f1_res(res_dict):
            new_dict = {}
            for k in res_dict:
                if isinstance(res_dict[k], dict):
                    new_dict.update(self.add_key_prefix(k, res_dict[k]))
                else:
                    new_dict[k] = res_dict[k]
            return new_dict

        aci_macro_f1_res = self.compute_macro_f1(aci_preds_list, aci_labels_list, 'aci')
        aci_macro_f1_res = post_process_macro_f1_res(aci_macro_f1_res)
        all_eval_res.update(self.add_key_prefix('aci_macro', aci_macro_f1_res))
        ari_macro_f1_res = self.compute_macro_f1(ari_preds_list, ari_labels_list, 'ari')
        ari_macro_f1_res = post_process_macro_f1_res(ari_macro_f1_res)
        all_eval_res.update(self.add_key_prefix('ari_macro', ari_macro_f1_res))

        return all_eval_res

    def cal_acl17_f1_score(self, aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list):
        all_eval_res = {}
        def add_ar(aci_labels_list, ari_labels_list, arict_labels_list):
            for aci_labels, ari_labels, arict_labels in zip(aci_labels_list, ari_labels_list, arict_labels_list):
                last_mc_id = [-1, -1]
                for node in reversed(aci_labels):
                    if 'MajorClaim' in node[1]:
                        last_mc_id = node[0]
                        break
                for node in aci_labels:
                    if 'MajorClaim' in node[1]:
                        ari_labels.append([node[0], "Root_rel", [0, 0]])
                        arict_labels.append([node[0], node[1], "Root_rel", [0, 0], "Root_node"])
                span2label = {}
                for node in aci_labels:
                    if node[1]!='Claim' and node[1].startswith('Claim'):
                        # if last_mc_id:
                        ari_labels.append([node[0], node[1].split(':')[1], last_mc_id])
                        arict_labels.append([node[0], node[1].split(':')[0], node[1].split(':')[1], last_mc_id, 'MajorClaim'])
                        node[1] = node[1].split(':')[0]
                    span2label[tuple(node[0])] = node[1]
                span2label[(0, 0)] = "Root_node"
                span2label[(-1, -1)] = "None_type_MC"
                
                for edge in arict_labels:
                    edge[1] = span2label[tuple(edge[0])]
                    edge[4] = span2label[tuple(edge[3])]
            return aci_labels_list, ari_labels_list, arict_labels_list
        
        aci_preds_list, ari_preds_list, arict_preds_list = add_ar(aci_preds_list, ari_preds_list, arict_preds_list)
        aci_labels_list, ari_labels_list, arict_labels_list = add_ar(aci_labels_list, ari_labels_list, arict_labels_list) 

        acl17_aci_micro_f1_res = self.am_f1_metric.compute(predictions=aci_preds_list, references=aci_labels_list)
        acl17_ari_micro_f1_res = self.am_f1_metric.compute(predictions=ari_preds_list, references=ari_labels_list)

        if arict_labels_list:
            acl17_arict_f1_res = self.am_f1_metric.compute(predictions=arict_preds_list, references=arict_labels_list)
        else:
            acl17_arict_f1_res = {}

        all_eval_res.update(self.add_key_prefix('acl17_aci_micro', acl17_aci_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('acl17_ari_micro', acl17_ari_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('acl17_e2e', acl17_arict_f1_res))

        macro_f1_eval_res = self.cal_macro_f1_score(aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list)
        all_eval_res.update(self.add_key_prefix('acl17', macro_f1_eval_res))

        return all_eval_res


    def cal_acl19_f1_score(self, aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list):
        all_eval_res = {}
        def transform_labels(aci_labels_list, ari_labels_list, arict_labels_list):
            for aci_labels, ari_labels, arict_labels in zip(aci_labels_list, ari_labels_list, arict_labels_list):

                span2label = {}
                for node in aci_labels:
                    if node[1]!='Claim' and node[1].startswith('Claim'):
                        node[1] = node[1].split(':')[0]
                    span2label[node[0]] = node[1]

                
                for edge in arict_labels:
                    edge[1] = span2label[edge[0]]
                    edge[4] = span2label[edge[3]]
            return aci_labels_list, ari_labels_list, arict_labels_list
        
        aci_preds_list, ari_preds_list, arict_preds_list = transform_labels(aci_preds_list, ari_preds_list, arict_preds_list)
        aci_labels_list, ari_labels_list, arict_labels_list = transform_labels(aci_labels_list, ari_labels_list, arict_labels_list) 

        acl19_aci_micro_f1_res = self.am_f1_metric.compute(predictions=aci_preds_list, references=aci_labels_list)
        acl19_ari_micro_f1_res = self.am_f1_metric.compute(predictions=ari_preds_list, references=ari_labels_list)

        if arict_labels_list:
            acl19_arict_f1_res = self.am_f1_metric.compute(predictions=arict_preds_list, references=arict_labels_list)
        else:
            acl19_arict_f1_res = {}

        all_eval_res.update(self.add_key_prefix('acl19_aci_micro', acl19_aci_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('acl19_ari_micro', acl19_ari_micro_f1_res))
        all_eval_res.update(self.add_key_prefix('acl19_arict', acl19_arict_f1_res))

        acl19_macro_f1_eval_res = self.cal_macro_f1_score(aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list)
        all_eval_res.update(self.add_key_prefix('acl19', acl19_macro_f1_eval_res))

        return all_eval_res



    def parse_target_seq(
            self, 
            source: str = None,
            target: str = None,
            ac_type_set: set = None,
            ar_type_set: set = None,
            view_type: str = None,
            orig_input_text: str = None,
            decoded_input_text: str = None,
            unk_replacements: str = None,
            task_type: str = None,
            use_oracle_span: bool = False,
            use_fuzzy_search: bool = False
        ):
        if task_type == 'AM':
            ari_pattern_1 = r'\[#(.*?)#\] is an? \"(.*?)\" type of argument for \[#(.*?)#\]'
            ari_pattern_2 = r', and is an? \"(.*?)\" type of argument for \[#(.*?)#\]'
            asa_pattern_1 = r'It is an? \"(.*?)\" type of argument for \[#(.*?)#\] \(\".*?\"\)'
            asa_pattern_2 = r', and is an? \"(.*?)\" type of argument for \[#(.*?)#\] \(\".*?\"\)'
        elif task_type == 'AQE':
            ari_pattern_1 = r'\[#(.*?)#\] is supported by the \"(.*?)\" evidence \[#(.*?)#\]'
            ari_pattern_2 = r', and is supported by the \"(.*?)\" evidence \[#(.*?)#\]'
            asa_pattern_1 = r'It is supported by the \"(.*?)\" evidence \[#(.*?)#\]'
            asa_pattern_2 = r', and is supported by the \"(.*?)\" evidence \[#(.*?)#\]'
            if os.environ.get('OLD_VERSION', 'false') == 'true':
                ari_pattern_1 = r'\[#(.*?)#\] is an? \"(.*?)\" type of evidence for \[#(.*?)#\]'
                ari_pattern_2 = r', and is an? \"(.*?)\" type of evidence for \[#(.*?)#\]'
                asa_pattern_1 = r'It is an? \"(.*?)\" type of evidence for \[#(.*?)#\]'
                asa_pattern_2 = r', and is an? \"(.*?)\" type of evidence for \[#(.*?)#\]'
        elif task_type == 'APE':
            ari_pattern_1 = r'\[#(.*?)#\] can form an \"(Argument-Pair)\" with \[#(.*?)#\]'
            ari_pattern_2 = r', and can form an \"(Argument-Pair)\" with \[#(.*?)#\]'
            asa_pattern_1 = r'It can form an \"(Argument-Pair)\" with \[#(.*?)#\]'
            asa_pattern_2 = r', and can form an \"(Argument-Pair)\" with \[#(.*?)#\]'
            if os.environ.get('OLD_VERSION', 'false') == 'true':
                ari_pattern_1 = r'\[#(.*?)#\] can form an (argument pair) with \[#(.*?)#\]'
                ari_pattern_2 = r', and can form an (argument pair) with \[#(.*?)#\]'
                asa_pattern_1 = r'It can form an (argument pair) with \[#(.*?)#\]'
                asa_pattern_2 = r', and can form an (argument pair) with \[#(.*?)#\]'

        def extract_aci_target(s):
            pattern = r'The type of \[# (.*?) #\] is \"(.*?)\".'
            matches = re.findall(pattern, s)
            assert len(matches) == 1
            parsed_list = list(matches[0])
            return parsed_list

        def extract_ari_target(s):
            none_ari_pattern = r'\[# (.*?) #\] produces no relation.'
            matches = re.findall(none_ari_pattern, s)
            if matches:
                assert len(matches) == 1
                parsed_list = [matches[0], []]
            else:
                matches = re.findall(ari_pattern_1, s)
                assert len(matches) == 1
                assert len(matches[0]) == 3
                matches[0] = [_.strip() for _ in matches[0]]
                ari_extracted_1 = [matches[0][0], [list(matches[0][1:])]]

                matches = re.findall(ari_pattern_2, s)
                ari_extracted_2 = [[_.strip() for _ in list(tup)] for tup in matches]

                ari_extracted_1[1].extend(ari_extracted_2)
                parsed_list = ari_extracted_1
            return parsed_list

        def extract_asa_target(s):
            aci_pattern = r'The type of \[# (.*?) #\] is \"(.*?)\".'
            matches = re.findall(aci_pattern, s)
            assert len(matches) == 1
            aci_extracted = list(matches[0])

            matches = re.findall(asa_pattern_1, s)
            assert len(matches) <= 1
            ari_extracted_1 = [[_.strip() for _ in list(tup)] for tup in matches]

            matches = re.findall(asa_pattern_2, s)
            ari_extracted_2 = [[_.strip() for _ in list(tup)] for tup in matches]

            ari_extracted = ari_extracted_1 + ari_extracted_2

            parsed_list = aci_extracted + [ari_extracted]
            return parsed_list

        sub_targets = target.split("[SEP]")
        parsed_lists = []

        for sub_target in sub_targets:
            if sub_target.startswith('[ACI]') or sub_target.startswith('[ARI]') or sub_target.startswith('[ASA]'):
                sub_target = sub_target[5:]
            sub_target = sub_target.strip()
            if not sub_target:
                continue
            if sub_target == 'None':
                continue
            parsed_list = None
            if view_type == '[ACI]':
                try:
                    parsed_list = extract_aci_target(sub_target)
                except:
                    logger.warning(f"Target Parsing Warning 0: Failed to extract_aci_target from {{{sub_target}}}.")
                    continue
            elif view_type == '[ARI]':
                try:
                    parsed_list = extract_ari_target(sub_target)
                except:
                    logger.warning(f"Target Parsing Warning 1: Failed to extract_ari_target from {{{sub_target}}}.")
                    continue
            elif view_type == '[ASA]':
                try:
                    parsed_list = extract_asa_target(sub_target)
                except:
                    logger.warning(f"Target Parsing Warning 2: Failed to extract_asa_target from {{{sub_target}}}.")
                    continue
            else:
                raise ValueError(f"Unknown prefix in orig_input_text: {{{orig_input_text}}}")
            if parsed_list:
                parsed_lists.append(parsed_list)

        aci_res_list, ari_res_list = [], []
        if task_type == "AM":
            if use_oracle_span:
                for parsed_list in parsed_lists:

                    if view_type == '[ACI]':
                        ac_idx, ac_type = parsed_list
                        if ac_type_set and not ac_type in ac_type_set:
                            logger.warning(f"Target Parsing Warning 3: Unknown ac_type {{{ac_type}}} in ac_type_set: {{{ac_type_set}}}")
                            continue
                        if f'[# {ac_idx} #]' not in source:
                            logger.warning(f"Target Parsing Warning 4: ac_idx {{{ac_idx}}} not in source: {{{source}}}")
                            continue
                        aci_res_list.append((int(ac_idx), ac_type))
                    elif view_type == '[ARI]':
                        src_idx, tuple_list = parsed_list
                        for t in tuple_list:
                            rel_type, tgt_ac_idx = t
                            if not rel_type in ar_type_set:
                                logger.warning(f"Target Parsing Warning 5: Unknown rel_type {{{rel_type}}} in ar_type_set: {{{ar_type_set}}}")
                                continue
                            if f'[# {src_idx} #]' not in source:
                                logger.warning(f"Target Parsing Warning 6: src_idx {{{src_idx}}} not in source: {{{source}}}")
                                continue
                            if f'[# {tgt_ac_idx} #]' not in source:
                                logger.warning(f"Target Parsing Warning 7: tgt_ac_idx {{{tgt_ac_idx}}} not in source: {{{source}}}")
                                continue
                            ari_res_list.append((int(src_idx), rel_type, int(tgt_ac_idx)))
                    elif view_type == '[ASA]':
                        ac_idx, ac_type, tuple_list = parsed_list
                        if ac_type_set and not ac_type in ac_type_set:
                            logger.warning(f"Target Parsing Warning 8: Unknown ac_type {{{ac_type}}} in ac_type_set: {{{ac_type_set}}}")
                            continue
                        if f'[# {ac_idx} #]' not in source:
                            logger.warning(f"Target Parsing Warning 9: ac_idx {{{ac_idx}}} not in source: {{{source}}}")
                            continue
                        for t in tuple_list:
                            rel_type, tgt_ac_idx = t
                            if not rel_type in ar_type_set:
                                logger.warning(f"Target Parsing Warning 10: Unknown rel_type {{{rel_type}}} in ar_type_set: {{{ar_type_set}}}")
                                continue
                            if f'[# {tgt_ac_idx} #]' not in source:
                                logger.warning(f"Target Parsing Warning 11: tgt_ac_idx {{{tgt_ac_idx}}} not in source: {{{source}}}")
                                continue
                            ari_res_list.append((int(ac_idx), rel_type, int(tgt_ac_idx)))
                        aci_res_list.append((int(ac_idx), ac_type))
                    else:   
                        raise ValueError(f"Unknown prefix in source: {{{source}}}")
            else:
                ac_text_list = [_[0] for _ in parsed_lists]
                if use_fuzzy_search:
                    ac_spans, valid_ac_flags = get_spans_fuzzy(orig_input_text, ac_text_list)
                else:
                    ac_spans, valid_ac_flags = get_spans(decoded_input_text, ac_text_list)
                # use the valid_ac_flags to update parsed_dict_list
                parsed_lists = [_ for i, _ in enumerate(parsed_lists) if valid_ac_flags[i]]
                ac_text_list = [_[0] for _ in parsed_lists]
                assert len(parsed_lists) == len(ac_spans)

                ac_text_to_span = {ac_text: ac_span for ac_text, ac_span in zip(ac_text_list, ac_spans)}

                aci_res_list, ari_res_list = [], []
                for src_ac_span, parsed_list in zip(ac_spans, parsed_lists):
                    try:
                        if len(parsed_list) == 3:
                            ac_type = parsed_list[1]
                            ar_list = parsed_list[2]
                        elif len(parsed_list) == 2:
                            if isinstance(parsed_list[1], str):
                                ac_type = parsed_list[1]
                                ar_list = None
                            elif isinstance(parsed_list[1], list):
                                ac_type = None
                                ar_list = parsed_list[1]
                            else:
                                logger.warning(f"Target Parsing Warning 12: Failed to parse JSON object {parsed_list}.")
                                continue
                        else:
                            logger.warning(f"Target Parsing Warning 13: Failed to parse JSON object {parsed_list}.")
                            continue

                        if ac_type:
                            assert ac_type in ac_type_set
                            aci_res_list.append((tuple(src_ac_span), ac_type))

                        if ar_list:
                            for rel_type, tgt_ac_text in ar_list:
                                assert rel_type in ar_type_set
                                if tgt_ac_text not in ac_text_to_span:
                                    logger.warning(f"Target Parsing Warning 14: tgt_ac_text {{{tgt_ac_text}}} not in ac_text_to_span. parsed_list: {parsed_list}.")
                                    continue
                                ari_res_list.append((tuple(src_ac_span), rel_type, tuple(ac_text_to_span[tgt_ac_text])))
                    except:
                        logger.warning(f"Target Parsing Warning 15: Failed to parse JSON object {parsed_list}.")
                        continue
                
                if not use_fuzzy_search:
                    # replace unk token
                    def replace_within_span(text, span, replace_dict):
                        start, end = span
                        adjustments = 0  # Used to account for the changes in length due to replacements
                        # Sort the replace_dict by the start of the spans to ensure replacements are done in order
                        for (r_start, r_end), value in sorted(replace_dict.items(), key=lambda x: x[0][0]):
                            if r_start >= start and r_end <= end:
                                offset_start = r_start - start + adjustments
                                offset_end = r_end - start + adjustments
                                text = text[:offset_start] + value + text[offset_end:]
                                adjustments += len(value) - (r_end - r_start)
                        return text
                    unk_replacements = json.loads(unk_replacements)
                    unk_replacements = {ast.literal_eval(k): v for k, v in unk_replacements.items()}
                    if unk_replacements:
                        orig_ac_text_list = []
                        for ac_span, parsed_list in zip(ac_spans, parsed_lists):
                            ac_text = parsed_list[0]
                            ac_text = replace_within_span(ac_text, ac_span, unk_replacements)
                            orig_ac_text_list.append(ac_text)
                            parsed_list[0] = orig_ac_text_list
                        if use_fuzzy_search:
                            orig_ac_spans, orig_valid_ac_flags = get_spans_fuzzy(orig_input_text, orig_ac_text_list)
                        else:
                            orig_ac_spans, orig_valid_ac_flags = get_spans(orig_input_text, orig_ac_text_list)
                        # TODO: check why bug when using abstrct
                        assert all(orig_valid_ac_flags)
                        pos_adjust_dict = {tuple(ac_span): tuple(orig_ac_span) for ac_span, orig_ac_span in zip(ac_spans, orig_ac_spans)}

                        aci_res_list = [
                            (pos_adjust_dict[ac_span], ac_type) 
                            for ac_span, ac_type in aci_res_list
                            ]
                        ari_res_list = [
                            (pos_adjust_dict[ac_span], rel_type, pos_adjust_dict[tgt_ac_span]) 
                            for ac_span, rel_type, tgt_ac_span in ari_res_list
                            ]

        elif task_type == "APE":
            for parsed_list in parsed_lists:
                try:
                    if view_type == '[ACI]':
                        ac_idxs_str, ac_type = parsed_list
                        ac_idxs = ac_idxs_str.split(' ')
                        start_ac_idx = int(ac_idxs[0])
                        end_ac_idx = int(ac_idxs[-1])
                        if ac_idxs_str != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))):
                            logger.warning(f"Target Parsing Warning 3: ac_idxs_str {{{ac_idxs_str}}} != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))): {{{' '.join(map(str, range(start_ac_idx, end_ac_idx+1)))}}}")
                            continue
                        if f'[# {ac_idxs[0]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 4: ac_idx {{{ac_idxs[0]}}} not in source: {{{source}}}")
                            continue
                        if f'[# {ac_idxs[-1]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 5: ac_idx {{{ac_idxs[-1]}}} not in source: {{{source}}}")
                            continue
                        aci_res_list.append((start_ac_idx, end_ac_idx+1))
                    elif view_type == '[ARI]':
                        ac_idxs_str, tuple_list = parsed_list
                        ac_idxs = ac_idxs_str.split(' ')
                        start_ac_idx = int(ac_idxs[0])
                        end_ac_idx = int(ac_idxs[-1])
                        if ac_idxs_str != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))):
                            logger.warning(f"Target Parsing Warning 8: ac_idxs_str {{{ac_idxs_str}}} != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))): {{{' '.join(map(str, range(start_ac_idx, end_ac_idx+1)))}}}")
                            continue
                        if f'[# {ac_idxs[0]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 9: ac_idx {{{ac_idxs[0]}}} not in source: {{{source}}}")
                            continue
                        if f'[# {ac_idxs[-1]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 10: ac_idx {{{ac_idxs[-1]}}} not in source: {{{source}}}")
                            continue

                        for t in tuple_list:
                            tgt_ac_idxs_str = t[1]
                            tgt_ac_idxs = tgt_ac_idxs_str.split(' ')
                            start_tgt_ac_idx = int(tgt_ac_idxs[0])
                            end_tgt_ac_idx = int(tgt_ac_idxs[-1])
                            if tgt_ac_idxs_str != ' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1))):
                                logger.warning(f"Target Parsing Warning 11: tgt_ac_idxs_str {{{tgt_ac_idxs_str}}} != ' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1))): {{{' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1)))}}}")
                                continue
                            if f'[# {tgt_ac_idxs[0]} #]' not in source:
                                logger.warning(f"Target Parsing Warning 12: tgt_ac_idx {{{tgt_ac_idxs[0]}}} not in source: {{{source}}}")
                                continue
                            if f'[# {tgt_ac_idxs[-1]} #]' not in source:
                                logger.warning(f"Target Parsing Warning 13: tgt_ac_idx {{{tgt_ac_idxs[-1]}}} not in source: {{{source}}}")
                                continue
                            ari_res_list.append(((start_ac_idx, end_ac_idx+1), (start_tgt_ac_idx, end_tgt_ac_idx+1)))
                    elif view_type == '[ASA]':
                        ac_idxs_str, ac_type, tuple_list = parsed_list
                        ac_idxs = ac_idxs_str.split(' ')
                        start_ac_idx = int(ac_idxs[0])
                        end_ac_idx = int(ac_idxs[-1])
                        if ac_idxs_str != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))):
                            logger.warning(f"Target Parsing Warning 8: ac_idxs_str {{{ac_idxs_str}}} != ' '.join(map(str, range(start_ac_idx, end_ac_idx+1))): {{{' '.join(map(str, range(start_ac_idx, end_ac_idx+1)))}}}")
                            continue
                        if f'[# {ac_idxs[0]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 9: ac_idx {{{ac_idxs[0]}}} not in source: {{{source}}}")
                            continue
                        if f'[# {ac_idxs[-1]} #]' not in source:
                            logger.warning(f"Target Parsing Warning 10: ac_idx {{{ac_idxs[-1]}}} not in source: {{{source}}}")
                            continue
                        for t in tuple_list:
                            tgt_ac_idxs_str = t[1]
                            tgt_ac_idxs = tgt_ac_idxs_str.split(' ')
                            start_tgt_ac_idx = int(tgt_ac_idxs[0])
                            end_tgt_ac_idx = int(tgt_ac_idxs[-1])
                            if tgt_ac_idxs_str != ' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1))):
                                logger.warning(f"Target Parsing Warning 11: tgt_ac_idxs_str {{{tgt_ac_idxs_str}}} != ' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1))): {{{' '.join(map(str, range(start_tgt_ac_idx, end_tgt_ac_idx+1)))}}}")
                                continue
                            if f'[# {tgt_ac_idxs[0]} #]' not in source:
                                logger.warning(f"Target Parsing Warning 12: tgt_ac_idx {{{tgt_ac_idxs[0]}}} not in source: {{{source}}}")
                                continue
                            if f'[# {tgt_ac_idxs[-1]} #]' not in source:
                                logger.warning(f"Target Parsing Warning 13: tgt_ac_idx {{{tgt_ac_idxs[-1]}}} not in source: {{{source}}}")
                                continue
                            ari_res_list.append(((start_ac_idx, end_ac_idx+1), (start_tgt_ac_idx, end_tgt_ac_idx+1)))
                        aci_res_list.append((start_ac_idx, end_ac_idx+1))
                    else:   
                        raise ValueError(f"Unknown prefix in source: {{{source}}}")
                except:
                    logger.warning("Failed to parse target_seq.")
        elif task_type == "AQE":
            for parsed_list in parsed_lists:

                if view_type == '[ACI]':
                    ac_idx, ac_type = parsed_list
                    if not ac_type in ac_type_set:
                        logger.warning(f"Target Parsing Warning 3: Unknown ac_type {{{ac_type}}} in ac_type_set: {{{ac_type_set}}}")
                        continue
                    if f'[# {ac_idx} #]' not in source:
                        logger.warning(f"Target Parsing Warning 4: ac_idx {{{ac_idx}}} not in source: {{{source}}}")
                        continue
                    aci_res_list.append((int(ac_idx), ac_type))
                elif view_type == '[ARI]':
                    src_idx, tuple_list = parsed_list
                    for t in tuple_list:
                        rel_type, tgt_ac_idx = t
                        if not rel_type in ar_type_set:
                            logger.warning(f"Target Parsing Warning 5: Unknown rel_type {{{rel_type}}} in ar_type_set: {{{ar_type_set}}}")
                            continue
                        if f'[# {src_idx} #]' not in source:
                            logger.warning(f"Target Parsing Warning 6: src_idx {{{src_idx}}} not in source: {{{source}}}")
                            continue
                        if f'[# {tgt_ac_idx} #]' not in source:
                            logger.warning(f"Target Parsing Warning 7: tgt_ac_idx {{{tgt_ac_idx}}} not in source: {{{source}}}")
                            continue
                        ari_res_list.append((int(src_idx), rel_type, int(tgt_ac_idx)))
                elif view_type == '[ASA]':
                    ac_idx, ac_type, tuple_list = parsed_list
                    if not ac_type in ac_type_set:
                        logger.warning(f"Target Parsing Warning 8: Unknown ac_type {{{ac_type}}} in ac_type_set: {{{ac_type_set}}}")
                        continue
                    if f'[# {ac_idx} #]' not in source:
                        logger.warning(f"Target Parsing Warning 9: ac_idx {{{ac_idx}}} not in source: {{{source}}}")
                        continue
                    for t in tuple_list:
                        rel_type, tgt_ac_idx = t
                        if not rel_type in ar_type_set:
                            logger.warning(f"Target Parsing Warning 10: Unknown rel_type {{{rel_type}}} in ar_type_set: {{{ar_type_set}}}")
                            continue
                        if f'[# {tgt_ac_idx} #]' not in source:
                            logger.warning(f"Target Parsing Warning 11: tgt_ac_idx {{{tgt_ac_idx}}} not in source: {{{source}}}")
                            continue
                        ari_res_list.append((int(ac_idx), rel_type, int(tgt_ac_idx)))
                    aci_res_list.append((int(ac_idx), ac_type))
                else:   
                    raise ValueError(f"Unknown prefix in source: {{{source}}}")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
            
        # Deduplication
        aci_res_list = list(set([tuple(_) for _ in aci_res_list]))
        aci_res_list = sorted(aci_res_list)
        ari_res_list = list(set([tuple(_) for _ in ari_res_list]))
        ari_res_list = sorted(ari_res_list)
        return aci_res_list, ari_res_list


    def convert_tuples_to_lists(self, obj):
        """
        Recursively converts tuples in a (possibly nested) list to lists.
        """
        if isinstance(obj, tuple):
            return [self.convert_tuples_to_lists(item) for item in obj]
        elif isinstance(obj, list):
            return [self.convert_tuples_to_lists(item) for item in obj]
        return obj

    def get_label_and_pred_tuples(self):
        aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list, arict_labels_list = \
            self.get_labels_and_preds()

        # create a mapping dictionary from index to type for each sublist
        aci_preds_dict_list = [dict(sublist) for sublist in aci_preds_list]
        if self.data_args.task_type == "AQE":
            arict_preds_list = [
                # [
                #     [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                #     if item[0] in aci_preds_dict_list[i]
                #     else []
                #     for item in sublist
                # ]
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                    for item in sublist if item[0] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(ari_preds_list)
            ]
        elif self.data_args.task_type == "AM":
            arict_preds_list = [
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], aci_preds_dict_list[i][item[2]]]
                    for item in sublist if item[0] in aci_preds_dict_list[i] and item[2] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(ari_preds_list)
            ]
        else:
            arict_preds_list = []

        # change tuple into list
        aci_preds_list = self.convert_tuples_to_lists(aci_preds_list)
        ari_preds_list = self.convert_tuples_to_lists(ari_preds_list)

        return aci_labels_list, aci_preds_list, ari_labels_list, ari_preds_list, arict_labels_list, arict_preds_list


    def cal_score_for_each_view(self):


        def get_pred_labels(pred_aci_list, pred_ari_triple_list, sample_idx, voting_threshold):
            assert len(pred_aci_list) == len(pred_ari_triple_list) == len(sample_idx)

            pred_ac_from_aci_view = []
            pred_ar_from_ari_view = []
            pred_ac_from_asa_view = []
            pred_ar_from_asa_view = []

            for i, (idx, pred_aci, pred_ari_to) in enumerate(zip(sample_idx, pred_aci_list, pred_ari_triple_list)):
                if i % 3 == 0:
                    pred_ac_from_aci_view.append(pred_aci)
                elif i % 3 == 1:
                    pred_ar_from_ari_view.append(pred_ari_to)
                elif i % 3 == 2:
                    pred_ac_from_asa_view.append(pred_aci)
                    pred_ar_from_asa_view.append(pred_ari_to)
            
            assert len(pred_ac_from_aci_view) == len(pred_ar_from_ari_view) == len(pred_ac_from_asa_view) == len(pred_ar_from_asa_view)

            return pred_ac_from_aci_view, pred_ar_from_ari_view, pred_ac_from_asa_view, pred_ar_from_asa_view

        def get_labels_and_preds():
            sample_idx = self.eval_set['idx']

            aci_labels_list, ari_labels_list, arict_labels_list = self.get_true_labels()

            if self.data_args.task_type == 'AM':
                ac_type_set = set([_[-1] for sublist in aci_labels_list for _ in sublist])
                ar_type_set = set([_[1] for sublist in ari_labels_list for _ in sublist])
            elif self.data_args.task_type == 'APE':
                ac_type_set, ar_type_set = None, None
            elif self.data_args.task_type == 'AQE':
                ac_type_set = set([_[-1] for sublist in aci_labels_list for _ in sublist])
                ar_type_set = set([_[2] for sublist in ari_labels_list for _ in sublist])

            aci_preds_list = []
            ari_preds_list = []
            for (
                source,
                pred_target,
                view_type,
                orig_input_text,
                decoded_input_text,
                unk_replacements
            ) in zip(
                    self.eval_set['source'],
                    self.decoded_preds,
                    self.eval_set['view_type'],
                    self.eval_set['orig_input_text'],
                    self.eval_set['decoded_input_text'],
                    self.eval_set['unk_replacements']
                ):
                aci_preds, ari_preds = self.parse_target_seq(
                    source=source, 
                    target=pred_target,
                    ac_type_set=ac_type_set,
                    ar_type_set=ar_type_set,
                    view_type=view_type,
                    orig_input_text=orig_input_text,
                    decoded_input_text=decoded_input_text,
                    unk_replacements=unk_replacements,
                    task_type=self.data_args.task_type,
                    use_oracle_span=self.data_args.use_oracle_span,
                    use_fuzzy_search=self.data_args.use_fuzzy_search
                )
                aci_preds_list.append(aci_preds)
                ari_preds_list.append(ari_preds)

            pred_ac_from_aci_view, pred_ar_from_ari_view, pred_ac_from_asa_view, pred_ar_from_asa_view = get_pred_labels(
                aci_preds_list, ari_preds_list, sample_idx, self.data_args.voting_threshold
            )
            return pred_ac_from_aci_view, pred_ar_from_ari_view, pred_ac_from_asa_view, pred_ar_from_asa_view, aci_labels_list, ari_labels_list, arict_labels_list
        

        pred_ac_from_aci_view, pred_ar_from_ari_view, pred_ac_from_asa_view, pred_ar_from_asa_view, aci_labels_list, ari_labels_list, arict_labels_list = \
            get_labels_and_preds()

        # create a mapping dictionary from index to type for each sublist
        aci_preds_dict_list = [dict(sublist) for sublist in pred_ac_from_asa_view]
        if self.data_args.task_type == "AQE":
            arict_preds_list = [
                # [
                #     [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                #     if item[0] in aci_preds_dict_list[i]
                #     else []
                #     for item in sublist
                # ]
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], 'dummy']
                    for item in sublist if item[0] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(pred_ar_from_asa_view)
            ]
            assert arict_labels_list == None
            arict_labels_list = ari_labels_list
            ari_labels_list = [[[__[0], __[2], __[3]] for __ in _] for _ in ari_labels_list]
        elif self.data_args.task_type == "AM":
            arict_preds_list = [
                [
                    [item[0], aci_preds_dict_list[i][item[0]], item[1], item[2], aci_preds_dict_list[i][item[2]]]
                    for item in sublist if item[0] in aci_preds_dict_list[i] and item[2] in aci_preds_dict_list[i]
                ]
                for i, sublist in enumerate(pred_ar_from_asa_view)
            ]
        else:
            arict_preds_list = []

        # change tuple into list
        pred_ac_from_aci_view = self.convert_tuples_to_lists(pred_ac_from_aci_view)
        pred_ar_from_ari_view = self.convert_tuples_to_lists(pred_ar_from_ari_view)
        pred_ac_from_asa_view = self.convert_tuples_to_lists(pred_ac_from_asa_view)
        pred_ar_from_asa_view = self.convert_tuples_to_lists(pred_ar_from_asa_view)

        all_eval_res = {}

        aci_micro_f1_res_from_aci_view = self.am_f1_metric.compute(predictions=pred_ac_from_aci_view, references=aci_labels_list)
        aci_micro_f1_res_from_asa_view = self.am_f1_metric.compute(predictions=pred_ac_from_asa_view, references=aci_labels_list)

        if self.data_args.task_type == 'AQE':
            # Need to be fix
            ari_micro_f1_res_from_ari_view = self.am_f1_metric.compute(predictions=pred_ar_from_ari_view, references=ari_labels_list)
            ari_micro_f1_res_from_asa_view = self.am_f1_metric.compute(predictions=pred_ar_from_asa_view, references=ari_labels_list)
        else:
            ari_micro_f1_res_from_ari_view = self.am_f1_metric.compute(predictions=pred_ar_from_ari_view, references=ari_labels_list)
            ari_micro_f1_res_from_asa_view = self.am_f1_metric.compute(predictions=pred_ar_from_asa_view, references=ari_labels_list)

        if arict_labels_list:
            arict_preds_list = self.convert_tuples_to_lists(arict_preds_list)
            arict_f1_res = self.am_f1_metric.compute(predictions=arict_preds_list, references=arict_labels_list)
        else:
            arict_f1_res = {}
        # aci_micro_f1_F, aci_micro_f1_P, aci_micro_f1_R
        all_eval_res.update(self.add_key_prefix('aci_micro_from_aci_view', aci_micro_f1_res_from_aci_view))
        all_eval_res.update(self.add_key_prefix('aci_micro_from_asa_view', aci_micro_f1_res_from_asa_view))
        all_eval_res.update(self.add_key_prefix('ari_micro_from_ari_view', ari_micro_f1_res_from_ari_view))
        all_eval_res.update(self.add_key_prefix('ari_micro_from_asa_view', ari_micro_f1_res_from_asa_view))
        all_eval_res.update(self.add_key_prefix('arict_from_asa_view', arict_f1_res))
        
        # if self.data_args.task_type != 'AM':
        #     return all_eval_res
        # if self.data_args.task_type == 'AQE':
        #     all_eval_res['aqe_f1'] = all_eval_res['ari_micro_f1']
        #     all_eval_res['aqe_precision'] = all_eval_res['ari_micro_precision']
        #     all_eval_res['aqe_recall'] = all_eval_res['ari_micro_recall']
        # elif self.data_args.task_type == 'APE':
        #     all_eval_res['ape_f1'] = all_eval_res['ari_micro_f1']
        #     all_eval_res['ape_precision'] = all_eval_res['ari_micro_precision']
        #     all_eval_res['ape_recall'] = all_eval_res['ari_micro_recall']
        # elif self.data_args.task_type == 'AM':
        #     all_eval_res.update(self.cal_macro_f1_score(aci_preds_list, ari_preds_list, aci_labels_list, ari_labels_list))
        #     if not self.data_args.use_oracle_span:
        #         all_eval_res.update(self.cal_acl17_f1_score(aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list))
        #     else:
        #         file_path = self.data_args.test_file if self.data_args.test_file else self.data_args.train_file
        #         if 'aaec_paragraph_level' in file_path:
        #             all_eval_res.update(self.cal_acl19_f1_score(aci_preds_list, ari_preds_list, arict_preds_list, aci_labels_list, ari_labels_list, arict_labels_list))
        return all_eval_res
    