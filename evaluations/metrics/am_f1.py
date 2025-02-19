"""Calculate F1 scores for argument mining"""

import evaluate
import datasets


_CITATION = """\
None.
"""

_DESCRIPTION = """\
Span-level F1 scores.
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of list of tuples. 
    references: list of reference for each prediction. Each reference should be a list of tuples.
Returns:
    f1: am F1 score,
    precision: am precision score,
    recall: am recall score,
Examples:
    >>> am_f1 = evaluate.load("am_f1.py")
    >>> results = am_f1.compute(references=[[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]], predictions=[[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    >>> print(results)
    {'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SpanF1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="Metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    "references": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                }
            ),
        )
    def _compute(self, predictions, references):
        """Returns the scores"""
        # predictions and references are lists of lists of tuples (each tuple is also essentially a list)
        tp, fp, fn = 0, 0, 0
        # turn predictions and references into sets of tuples
        predictions = [set(map(tuple, prediction)) for prediction in predictions]
        references = [set(map(tuple, reference)) for reference in references]
        for prediction, reference in zip(predictions, references):
            tp += len(set(prediction) & set(reference))
            fp += len(set(prediction) - set(reference))
            fn += len(set(reference) - set(prediction))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            "f1": f1_score,
            "precision": precision,
            "recall": recall,
        }