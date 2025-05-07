from typing import Tuple

import torch
from loguru import logger
from PIL import Image
from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel, ViTImageProcessor


def load_model(model_name: str) -> Tuple[VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast, torch.device]:
    """load pretrained model, image processor and tokenizer from huggingface hub

    Args:
        model_name (str): Trained model name on huggingface hub

    Returns:
        Tuple[VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast, torch.device]: The loaded model, image processor, tokenizer, and device.
    """
    # Not using MPS as some of the operations are not supported.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load the image processor and tokenizer
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    # Load the model
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    return model, image_processor, tokenizer, device


def predict_caption(
    model: VisionEncoderDecoderModel,
    image_processor: ViTImageProcessor,
    tokenizer: GPT2TokenizerFast,
    device: torch.device,
    image_path: str,
) -> str:
    """Predict the caption for a given image using the loaded model.

    Args:
        model (VisionEncoderDecoderModel): The loaded model.
        image_processor (ViTImageProcessor): The image processor.
        tokenizer (GPT2TokenizerFast): The tokenizer.
        device (torch.device): The device to run the model on.
        image_path (str): The path to the image.

    Returns:
        str: The predicted caption.
    """
    # Load and process the image
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt", padding=True).pixel_values.to(device)

    # Generate the caption
    generated_caption = tokenizer.decode(
        model.generate(
            pixel_values,
            max_length=100,
            num_beams=5,
            do_sample=True,
            top_k=50,
            temperature=0.9,
        )[0],
        skip_special_tokens=True,
    )

    return generated_caption
