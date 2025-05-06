from typing import Any, Callable, Dict, Optional

import torch
from datasets import Dataset as HFDataset
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, ViTImageProcessor


class Flickr30kDataset(Dataset):
    """
    A custom dataset class for processing the Flickr30k dataset,
    designed for fine-tuning models in image captioning tasks.

    Args:
        dataset (HFDataset): A dataset object containing image-caption pairs,
            where each item is a dictionary with keys:
            - "image" (PIL.Image.Image): The input image.
            - "caption" (list[str]): A list of captions for the image.
        image_processor (ViTImageProcessor): A pre-trained image processor for processing
            images into model-compatible formats, such as pixel values.
        tokenizer (GPT2TokenizerFast): A pre-trained tokenizer for processing captions
            into tokenized sequences.
        transform (Optional[Callable[[Image.Image], Image.Image]]): A transformation
            function to apply to images before processing with the image_processor. Defaults to None.
        max_length (int): The maximum length for tokenized captions. Captions longer
            than this will be truncated, and shorter captions will be padded. Defaults to 50.

    Methods:
        __len__() -> int:
            Returns the total number of samples in the dataset.

        __getitem__(idx: int) -> Dict[str, torch.Tensor]:
            Processes the image-caption pair at the given index and returns a dictionary.

            Args:
                idx (int): The index of the sample to retrieve.

            Returns:
                Dict[str, torch.Tensor]: A dictionary containing:
                    - "pixel_values" (torch.Tensor): The processed image tensor.
                    - "labels" (torch.Tensor): The tokenized and padded/truncated caption tensor.
    """

    def __init__(
        self,
        dataset: HFDataset,
        image_processor: ViTImageProcessor,
        tokenizer: GPT2TokenizerFast,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        max_length: int = 50,
    ) -> None:
        self.dataset = dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        logger.info(f"Flickr30kDataset initialized with {len(self.dataset)} samples.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        try:
            # Process image and the first caption
            example = self.dataset[idx]
            image = example["image"]
            caption = example["caption"][0]

            # Apply image transformations if provided
            if self.transform:
                image = self.transform(image)

            # Process the image with the image processor
            pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze()

            # Tokenize the caption
            tokenized_caption = self.tokenizer(
                caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
            )

            labels = tokenized_caption["input_ids"].squeeze()

            return {
                "pixel_values": pixel_values,
                "labels": labels,
            }
        except Exception as e:
            logger.error(f"Error processing sample at index {idx}: {e}")
            raise RuntimeError(
                f"Failed to process sample at index {idx}. Check the dataset or processing logic."
            ) from e
