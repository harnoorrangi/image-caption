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

from imagecaption.scripts.make_dataset import Flickr30kDataset
from imagecaption.scripts.utils import compute_metrics, load_config


def get_and_prepare_data(
    dataset_name: str, tokenizer_gpt: GPT2TokenizerFast, image_processor_vit: ViTImageProcessor
) -> tuple:
    """
    Downloads and prepares the Flickr30k dataset for training, validation, and testing.

    Args:
        dataset_name (str): The name of the dataset to load (e.g., "flickr30k").
        tokenizer_gpt (GPT2TokenizerFast): A GPT-2 tokenizer for processing captions.
        image_processor_vit (ViTImageProcessor): A ViT image processor for processing images.

    Returns:
        tuple: A tuple containing:
            - train_dataset (Flickr30kDataset): Transformed training dataset.
            - val_dataset (Flickr30kDataset): Transformed validation dataset.
            - test_dataset (Flickr30kDataset): Transformed test dataset.
    """
    logger.info(f"Starting data preparation for dataset: {dataset_name}")
    # Download train, val and test dataset
    train_data = load_dataset(dataset_name, split="train")
    val_data = load_dataset(dataset_name, split="val")
    test_data = load_dataset(dataset_name, split="test")
    logger.success("Datasets downloaded successfully!")

    # Log dataset details
    logger.info(f"Train set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")

    # Transform train,valid and test dataset
    logger.info("Transforming datasets...")
    train_dataset = Flickr30kDataset(train_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
    val_dataset = Flickr30kDataset(val_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
    test_dataset = Flickr30kDataset(test_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
    logger.success("Datasets transformed successfully!")

    logger.info("Printing dimensions of final transformed dataset ")
    for item in train_dataset:
        logger.info(item["labels"].shape)
        logger.info(item["pixel_values"].shape)

    return train_dataset, val_dataset, test_dataset


def load_preprocessors(image_processor_vit: str, tokenizer_gpt: str) -> tuple:
    """
    Loads and initializes the Vision Transformer (ViT) image processor and GPT-2 tokenizer.

    Args:
        image_processor_vit (str): Path or model identifier for the ViT image processor.
        tokenizer_gpt (str): Path or model identifier for the GPT-2 tokenizer.

    Returns:
        tuple: A tuple containing:
            - image_processor (ViTImageProcessor): Initialized ViT image processor.
            - tokenizer (GPT2TokenizerFast): Initialized GPT-2 tokenizer.
    """
    # Load VIT Image processor and GPT2 Tokenizer
    image_processor = ViTImageProcessor.from_pretrained(image_processor_vit)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_gpt)
    # gpt2 does not have pad_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return image_processor, tokenizer


def initialize_model(
    encoder: str, decoder: str, max_length: int, early_stoping: bool, no_repeat_ngram: int, num_beans: int
) -> VisionEncoderDecoderModel:
    """
    Initializes a VisionEncoderDecoderModel with specific configurations.

    Args:
        encoder (str): The pre-trained encoder model (e.g., "google/vit-base-patch16-224").
        decoder (str): The pre-trained decoder model (e.g., "gpt2").
        max_length (int): The maximum length of generated sequences.
        early_stoping (bool): Whether to stop generation early when an EOS token is encountered.
        no_repeat_ngram (int): The size of the n-gram constraint to avoid repetition.
        num_beans (int): The number of beams to use in beam search.

    Returns:
        VisionEncoderDecoderModel: Configured vision-encoder-decoder model.
    """
    # Initialize model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder).to(device)

    # Model configuration for caption generation
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = max_length
    model.config.early_stopping = early_stoping
    model.config.no_repeat_ngram_size = no_repeat_ngram
    model.config.num_beams = num_beans

    return model


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead")
    # Load configuration yaml file
    config = load_config("configs/vit_gpt2.yaml")
    # Load VIT Image processor and GPT2 Tokenizer
    image_processor, tokenizer = load_preprocessors(
        image_processor_vit=config["encoder_model"], tokenizer_gpt=config["decoder_model"]
    )
    # Prepare dataset
    train_dataset, val_dataset, test_dataset = get_and_prepare_data(
        config["dataset"], tokenizer_gpt=tokenizer, image_processor_vit=image_processor
    )

    # Initialize model
    model = initialize_model(
        encoder=config["encoder_model"],
        decoder=config["decoder_model"],
        max_length=config["generation_max_length"],
        early_stoping=config["early_stopping_enabled"],
        no_repeat_ngram=config["no_repeat_ngram_size"],
        num_beans=config["num_beans"],
    )

    # Setup training using Hugging face
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["trained_model"],
        per_device_train_batch_size=config["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=config["VAL_BATCH_SIZE"],
        predict_with_generate=True,
        generation_max_length=config["generation_max_length"],
        generation_num_beams=config["num_beans"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1024,
        num_train_epochs=config.EPOCHS,
        report_to="none",
        push_to_hub=True,
        hub_model_id=config["hugging_face_model_id"],
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

    # train the model
    trainer.train()
