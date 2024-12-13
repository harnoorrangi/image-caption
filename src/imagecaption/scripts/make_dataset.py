from torch.utils.data import Dataset
import torch
from PIL import Image
from typing import Callable, Optional, Dict, Any


class Flickr30kDataset(Dataset):
    """
    A custom dataset class for processing the Flickr30k dataset,
    designed for fine-tuning models in image captioning tasks.

    Args:
        dataset (Dataset): A dataset object containing image-caption pairs,
            where each item is a dictionary with keys:
            - "image" (PIL.Image.Image): The input image.
            - "caption" (list of str): A list of captions for the image.
        image_processor (Callable[[Image.Image], Any]): A callable (e.g., a feature extractor)
            for processing images into model-compatible formats, such as pixel values.
        tokenizer (Callable[[str], Dict[str, Any]]): A tokenizer for processing captions into
            tokenized sequences.
        transform (Optional[Callable[[Image.Image], Image.Image]], optional): A transformation
            function to apply to images before processing with the image_processor. Defaults to None.
        max_length (int, optional): The maximum length for tokenized captions. Captions longer
            than this will be truncated, and shorter captions will be padded. Defaults to 50.

    Methods:
        __len__() -> int:
            Returns the total number of samples in the dataset.

        __getitem__(idx: int) -> Dict[str, Any]:
            Processes the image-caption pair at the given index and returns a dictionary.

            Args:
                idx (int): The index of the sample to retrieve.

            Returns:
                Dict[str, Any]: A dictionary containing:
                    - "pixel_values" (torch.Tensor): The processed image tensor.
                    - "labels" (torch.Tensor): The tokenized and padded/truncated caption tensor.

    Returns:
        An object of `Flickr30kDataset` that can be used for data loading in PyTorch models.
    """

    def __init__(self, dataset, image_processor, tokenizer, transform=None, max_length=50):
        self.dataset = dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Process image and the first caption
        example = self.dataset[idx]
        image = example["image"]
        caption = example["caption"][0]

        # Apply image transformations if provided
        if self.transform:
            image = self.transform(image)

        # Process the image with the image processor
        pixel_values = self.image_processor(
            images=image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize the caption
        tokenized_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = tokenized_caption["input_ids"].squeeze()

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
