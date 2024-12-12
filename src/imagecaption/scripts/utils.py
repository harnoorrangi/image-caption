from evaluate import load
import yaml
import os


def load_config(filename):
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return None

    try:
        with open(filename, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


# load the rouge and bleu metrics
rouge = load("rouge")
bleu = load("bleu")


def compute_metrics(eval_pred):
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

    return {
        **rouge_result,
        "bleu": bleu_result["bleu"] * 100,
        "gen_len": generation_length
    }
