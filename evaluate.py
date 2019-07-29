#!/usr/bin/env python
"""Computes the evaluation metrics
The metrics are computed using the coco-caption module
"""

import sys
import argparse

sys.path.append('coco-caption')
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def score_from_files(prediction_path, reference_path):
    # this is the generated captions
    hypo = {int(x[0]): [x[1].strip()] for x in
            (line.split('\t') for _, line in enumerate(open(prediction_path, 'r')))}

    # this is the ground truth captions
    refs = {}
    for _, line in enumerate(open(reference_path, 'r')):
        row = line.split('\t')
        idx = int(row[0])
        if idx in refs:
            refs[idx].append(row[1].strip())
        else:
            refs[idx] = [row[1].strip()]

    return score(refs, hypo)


def compute_scores(ref, hypo, metric_names):
    scorers = []
    if 'Bleu' in metric_names:
        scorers.append((Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]))
    if 'METEOR' in metric_names:
        scorers.append((Meteor(), "METEOR"))
    if 'ROUGE_L' in metric_names:
        scorers.append((Rouge(), "ROUGE_L"))
    if 'CIDEr' in metric_names:
        scorers.append((Cider(), "CIDEr"))
    if 'SPICE' in metric_names:
        scorers.append((Spice(), "SPICE"))

    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Evaluate the predicted descriptions')
    parser.add_argument('-prediction_path', type=str, help='path to file with the predicted descriptions to evaluate',
                        required=True)
    parser.add_argument('-reference_path', type=str, help='path to the file with the reference descriptions',
                        required=True)

    predictions = parser.prediction_path
    references = parser.reference_path

    score_from_files(prediction_path=predictions, reference_path=references)
