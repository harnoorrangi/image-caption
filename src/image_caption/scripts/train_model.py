from typing import Tuple

import torch
from datasets import load_dataset
from loguru import logger
from transformers import (
    GPT2TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    default_data_collator,
)
from image_caption.scripts.config import TrainingConfig
from image_caption.scripts.make_dataset import Flickr30kDataset
from image_caption.scripts.utils import compute_metrics



def get_and_prepare_data(
    dataset_name: str, tokenizer_gpt: GPT2TokenizerFast, image_processor_vit: ViTImageProcessor
) -> Tuple[Flickr30kDataset, Flickr30kDataset, Flickr30kDataset]:
    logger.info(f"Preparing dataset: {dataset_name}")

    train_data = load_dataset(dataset_name, split="train")
    val_data = load_dataset(dataset_name, split="val")
    test_data = load_dataset(dataset_name, split="test")

    train_dataset = Flickr30kDataset(train_data, tokenizer_gpt, image_processor_vit)
    val_dataset = Flickr30kDataset(val_data, tokenizer_gpt, image_processor_vit)
    test_dataset = Flickr30kDataset(test_data, tokenizer_gpt, image_processor_vit)

    return train_dataset, val_dataset, test_dataset


def load_preprocessors(image_processor_vit: str, tokenizer_gpt: str):
    image_processor = ViTImageProcessor.from_pretrained(image_processor_vit)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_gpt)
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def initialize_model(
    encoder: str, decoder: str, max_length: int, early_stopping: bool, no_repeat_ngram: int, num_beams: int
):
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)

    tokenizer = GPT2TokenizerFast.from_pretrained(decoder)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = max_length
    model.config.early_stopping = early_stopping
    model.config.no_repeat_ngram_size = no_repeat_ngram
    model.config.num_beams = num_beams

    return model



def train_main(cfg: TrainingConfig):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    image_processor, tokenizer = load_preprocessors(
        cfg.model_params.encoder_model, cfg.model_params.decoder_model
    )

    train_dataset, val_dataset, _ = get_and_prepare_data(
        cfg.dataset_params.dataset, tokenizer, image_processor
    )

    model = initialize_model(
        encoder=cfg.model_params.encoder_model,
        decoder=cfg.model_params.decoder_model,
        max_length=cfg.train_params.generation_max_length,
        early_stopping=cfg.train_params.early_stopping_enabled,
        no_repeat_ngram=cfg.train_params.no_repeat_ngram_size,
        num_beams=cfg.train_params.num_beans,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.model_params.trained_model,
        per_device_train_batch_size=cfg.train_params.train_batch_size,
        per_device_eval_batch_size=cfg.train_params.val_batch_size,
        predict_with_generate=True,
        generation_max_length=cfg.train_params.generation_max_length,
        generation_num_beams=cfg.train_params.num_beans,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1024,
        num_train_epochs=cfg.train_params.epochs,
        report_to="none",
        push_to_hub=True,
        hub_model_id=cfg.train_params.hugging_face_model_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    logger.success("Training completed successfully!")

