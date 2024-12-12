from torch.utils.data import Dataset
from PIL import Image


class Flickr30kDataset(Dataset):
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
