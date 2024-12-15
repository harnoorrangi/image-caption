from typing import Tuple
from typing_extensions import Annotated

import torch
import typer
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

app = typer.Typer()


def get_and_prepare_data(
    dataset_name: str, tokenizer_gpt: GPT2TokenizerFast, image_processor_vit: ViTImageProcessor
) -> Tuple[Flickr30kDataset, Flickr30kDataset, Flickr30kDataset]:
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

    try:
        # Download train, val and test dataset
        train_data = load_dataset(dataset_name, split="train")
        val_data = load_dataset(dataset_name, split="val")
        test_data = load_dataset(dataset_name, split="test")
        logger.success("Datasets downloaded successfully!")
    except Exception as e:
        logger.error(f"Error downloading {dataset_name}:{e} ")
        raise

    # Log dataset details
    logger.info(f"Train set size: {len(train_data)}")
    logger.info(f"Validation set size: {len(val_data)}")
    logger.info(f"Test set size: {len(test_data)}")

    # Transform train,valid and test dataset
    logger.info("Transforming datasets...")
    try:
        train_dataset = Flickr30kDataset(
            train_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
        val_dataset = Flickr30kDataset(
            val_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
        test_dataset = Flickr30kDataset(
            test_data, tokenizer=tokenizer_gpt, image_processor=image_processor_vit)
        logger.success("Datasets transformed successfully!")
    except Exception as e:
        logger.error(f"Error transforming datasets: {e}")
        raise

    logger.info("Printing dimensions of final transformed dataset ")
    for item in train_dataset:
        logger.info(item["labels"].shape)
        logger.info(item["pixel_values"].shape)

    return train_dataset, val_dataset, test_dataset


def load_preprocessors(image_processor_vit: str, tokenizer_gpt: str) -> Tuple[ViTImageProcessor, GPT2TokenizerFast]:
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
    logger.info(
        f"Loading processors: Image Processor (ViT) from {image_processor_vit}, Tokenizer (GPT-2) from {tokenizer_gpt}")
    try:
        # Load ViT Image Processor
        image_processor = ViTImageProcessor.from_pretrained(
            image_processor_vit)
        logger.success(
            f"Successfully loaded Image Processor from {image_processor_vit}")

        # Load GPT-2 Tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_gpt)
        logger.success(f"Successfully loaded Tokenizer from {tokenizer_gpt}")

        # Adjust GPT-2 tokenizer to have a pad_token
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    except Exception as e:
        logger.error(f"Error loading preprocessors: {e}")
        raise RuntimeError("Failed to load preprocessors") from e

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
    logger.info(
        f"Initializing VisionEncoderDecoderModel with encoder: {encoder}, decoder: {decoder}")
    logger.info(f"Configuration: max_length={max_length}, early_stopping={early_stoping}, "
                f"no_repeat_ngram_size={no_repeat_ngram}, num_beams={num_beans}")

    try:
        # Initialize the model with the pre-trained encoder and decoder
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder, decoder
        )
        logger.success("Model initialized successfully!")

        # Model configuration for caption generation
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.max_length = max_length
        model.config.early_stopping = early_stoping
        model.config.no_repeat_ngram_size = no_repeat_ngram
        model.config.num_beams = num_beans
        logger.success("Model configuration set successfully")

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise RuntimeError("Failed to initialize model") from e

    return model


@app.command()
def main(path: str = typer.Option(..., help="Path to the config file")):
    """Runs CLI to start fine tuning of the model

    Args:
        path (str, optional): _description_. Defaults to typer.Option(..., help="Path to the config file").

    Raises:
        RuntimeError: _description_
    """
    try:
        # Check for GPU availability
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"# available GPUs: {torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using the CPU instead")
        logger.debug(f"Path:{path}")
        # Load configuration from YAML file
        config = load_config(f"{path}")
        logger.success("Configuration loaded successfully")

        # Load VIT Image processor and GPT2 Tokenizer
        image_processor, tokenizer = load_preprocessors(
            image_processor_vit=config["encoder_model"], tokenizer_gpt=config["decoder_model"]
        )

        logger.success(
            "Preprocessors (image processor and tokenizer) loaded successfully")

        # Prepare dataset
        logger.info(f"Preparing dataset: {config['dataset']}")
        train_dataset, val_dataset, test_dataset = get_and_prepare_data(
            config["dataset"], tokenizer_gpt=tokenizer, image_processor_vit=image_processor
        )
        logger.success("Dataset preparation completed successfully")

        # Initialize model
        logger.info("Initializing model with configuration settings...")
        model = initialize_model(
            encoder=config["encoder_model"],
            decoder=config["decoder_model"],
            max_length=config["generation_max_length"],
            early_stoping=config["early_stopping_enabled"],
            no_repeat_ngram=config["no_repeat_ngram_size"],
            num_beans=config["num_beans"],
        )
        logger.success("Model initialized successfully")

        # Setup training using Hugging face
        logger.info("Setting up training arguments...")
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
        logger.info("Training arguments setup completed")

        logger.info("Initializing Seq2SeqTrainer...")
        trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=image_processor,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        logger.info("Trainer initialized successfully")

        # train the model
        logger.info("Starting model finetuning...")
        trainer.train()
        logger.success("Model finetuned completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during the execution: {e}")
        raise RuntimeError("Execution failed") from e


if __name__ == "__main__":
    # try:
    #     # Check for GPU availability
    #     if torch.cuda.is_available():
    #         device = torch.device("cuda")
    #         logger.info(f"# available GPUs: {torch.cuda.device_count()}")
    #     else:
    #         device = torch.device("cpu")
    #         logger.info("No GPU available, using the CPU instead")

    #     # Load configuration from YAML file
    #     config = load_config("configs/vit_gpt2.yaml")
    #     logger.success("Configuration loaded successfully")

    #     # Load VIT Image processor and GPT2 Tokenizer
    #     image_processor, tokenizer = load_preprocessors(
    #         image_processor_vit=config["encoder_model"], tokenizer_gpt=config["decoder_model"]
    #     )

    #     logger.success(
    #         "Preprocessors (image processor and tokenizer) loaded successfully")

    #     # Prepare dataset
    #     logger.info(f"Preparing dataset: {config['dataset']}")
    #     train_dataset, val_dataset, test_dataset = get_and_prepare_data(
    #         config["dataset"], tokenizer_gpt=tokenizer, image_processor_vit=image_processor
    #     )
    #     logger.success("Dataset preparation completed successfully")

    #     # Initialize model
    #     logger.info("Initializing model with configuration settings...")
    #     model = initialize_model(
    #         encoder=config["encoder_model"],
    #         decoder=config["decoder_model"],
    #         max_length=config["generation_max_length"],
    #         early_stoping=config["early_stopping_enabled"],
    #         no_repeat_ngram=config["no_repeat_ngram_size"],
    #         num_beans=config["num_beans"],
    #     )
    #     logger.success("Model initialized successfully")

    #     # Setup training using Hugging face
    #     logger.info("Setting up training arguments...")
    #     training_args = Seq2SeqTrainingArguments(
    #         output_dir=config["trained_model"],
    #         per_device_train_batch_size=config["TRAIN_BATCH_SIZE"],
    #         per_device_eval_batch_size=config["VAL_BATCH_SIZE"],
    #         predict_with_generate=True,
    #         generation_max_length=config["generation_max_length"],
    #         generation_num_beams=config["num_beans"],
    #         evaluation_strategy="epoch",
    #         save_strategy="epoch",
    #         save_total_limit=3,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_loss",
    #         greater_is_better=False,
    #         logging_steps=1024,
    #         num_train_epochs=config.EPOCHS,
    #         report_to="none",
    #         push_to_hub=True,
    #         hub_model_id=config["hugging_face_model_id"],
    #     )
    #     logger.info("Training arguments setup completed")

    #     logger.info("Initializing Seq2SeqTrainer...")
    #     trainer = Seq2SeqTrainer(
    #         model=model,
    #         tokenizer=image_processor,
    #         args=training_args,
    #         compute_metrics=compute_metrics,
    #         train_dataset=train_dataset,
    #         eval_dataset=val_dataset,
    #         data_collator=default_data_collator,
    #     )
    #     logger.info("Trainer initialized successfully")

    #     # train the model
    #     logger.info("Starting model finetuning...")
    #     trainer.train()
    #     logger.success("Model finetuned completed successfully")

    # except Exception as e:
    #     logger.error(f"An error occurred during the execution: {e}")
    #     raise RuntimeError("Execution failed") from e
    # TODO: Rewrite APP
    app()
