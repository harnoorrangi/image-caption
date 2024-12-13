import os
from typing import Any, Dict, Optional

import yaml
from evaluate import load
from evaluate.evaluator import Evaluator
from transformers import PreTrainedTokenizer


def load_config(filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads a YAML configuration file.

    Args:
        filename (str): Path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the configuration data if the file is valid
        and can be parsed. Returns None if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return None

    try:
        with open(filename, "r") as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def compute_metrics(eval_pred: Evaluator.PredictionOutput) -> Dict[str, float]:
    """
    Computes ROUGE and BLEU metrics for evaluating predictions against references.

    Args:
        eval_pred (Evaluator.PredictionOutput): An object containing:
            - `predictions` (torch.Tensor or list): The predicted sequences.
            - `label_ids` (torch.Tensor or list): The reference sequences.

    Returns:
        Dict[str, float]: A dictionary containing:
            - ROUGE scores (keys: "rouge1", "rouge2", "rougeL", etc.), scaled to percentages.
            - "bleu" (float): The BLEU score, scaled to a percentage.
            - "gen_len" (float): The average length of the generated captions.
    """
    # load the rouge and bleu metrics
    rouge = load("rouge")
    bleu = load("bleu")

    labels = eval_pred.label_ids
    preds = eval_pred.predictions

    # Decode the predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute the ROUGE score
    rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
    rouge_result = {k: v * 100 for k, v in rouge_result.items()}

    # Compute the BLEU score
    bleu_result = bleu.compute(predictions=pred_str, references=labels_str)

    # Get the length of the generated captions
    generation_length = bleu_result["translation_length"] / len(preds)

    return {**rouge_result, "bleu": bleu_result["bleu"] * 100, "gen_len": generation_length}
