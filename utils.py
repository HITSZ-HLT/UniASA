import numpy as np
from transformers.utils import logging
from evaluations.am_evaluation import AsaEvaluation
logger = logging.get_logger(__name__)

def postprocess_text(preds):
    # remove all the <pad> tokens
    preds = [pred.replace('<pad>', '') for pred in preds]
    preds = [pred.strip() for pred in preds]
    # remove the final </s> token
    preds = [pred[:-4] if pred.endswith('</s>') else pred for pred in preds]
    preds = [pred.strip() for pred in preds]

    return preds


def determine_eval_set_type(decoded_preds, raw_datasets):
    for set_type in ('validation', 'test'):
        if set_type in raw_datasets and len(decoded_preds) == len(raw_datasets[set_type]):
            return set_type
    raise ValueError("Unknown evaluation set type, please check the dataset.")

    
def get_compute_metrics_fn(data_args, tokenizer, am_f1_metric, raw_datasets):
    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        # if isinstance(preds, tuple):
        #     preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, clean_up_tokenization_spaces=False)

        # Some simple post-processing
        decoded_preds = postprocess_text(decoded_preds)

        eval_set_type = determine_eval_set_type(decoded_preds, raw_datasets)
        eval_set = raw_datasets[eval_set_type]

        am_eval = AsaEvaluation(data_args, eval_set, decoded_preds, am_f1_metric)
        eval_res = am_eval.cal_score()
        
        eval_res['overall_micro_f1'] = np.mean([eval_res['aci_micro_f1'], 
                                                eval_res['ari_micro_f1']]).tolist()
        # (eval_res['aci_micro_f1_results']['f1'] + eval_res['ari_micro_f1_results']['f1']) / 2

        # merge the results 
        # result = {
        #     'overall_f1': (aci_eval_res['f1'] + ari_eval_res_triple['f1']) / 2,
        #     'aci_f1': aci_eval_res['f1'],
        #     'aci_precision': aci_eval_res['precision'],
        #     'aci_recall': aci_eval_res['recall'],
        #     'ari_f1_triple': ari_eval_res_triple['f1'],
        #     'ari_precision_triple': ari_eval_res_triple['precision'],
        #     'ari_recall_triple': ari_eval_res_triple['recall'],
        #     'ari_f1': ari_eval_res['f1'],
        #     'ari_precision': ari_eval_res['precision'],
        #     'ari_recall': ari_eval_res['recall'],
        # }
        return eval_res
    return compute_metrics

def get_preprocess_function_fn(data_args, tokenizer, source_seq_column, target_seq_column):
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[source_seq_column])):
            if examples[source_seq_column][i] and examples[target_seq_column][i]:
                inputs.append(examples[source_seq_column][i])
                targets.append(examples[target_seq_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function