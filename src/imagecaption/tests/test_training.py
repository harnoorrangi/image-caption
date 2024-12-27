import unittest
import hydra
from unittest.mock import patch, MagicMock
from transformers import GPT2TokenizerFast, ViTImageProcessor
from imagecaption.scripts.make_dataset import Flickr30kDataset
from imagecaption.scripts.train_model import (
    get_and_prepare_data,
    load_preprocessors,
    initialize_model,
)


class TestImageCaptioning(unittest.TestCase):

    # @patch("datasets.load_dataset")
    # @patch("imagecaption.scripts.make_dataset.Flickr30kDataset")
    # def test_get_and_prepare_data(self, mock_flickr_dataset, mock_load_dataset):
    #     # Mock dataset
    #     mock_train = MagicMock()
    #     mock_val = MagicMock()
    #     mock_test = MagicMock()
    #     mock_load_dataset.side_effect = [mock_train, mock_val, mock_test]

    #     # Mock Flickr30kDataset transformation
    #     mock_flickr_dataset.side_effect = [
    #         MagicMock(),
    #         MagicMock(),
    #         MagicMock(),
    #     ]

    #     # Mock tokenizer and image processor
    #     tokenizer_mock = MagicMock(spec=GPT2TokenizerFast)
    #     image_processor_mock = MagicMock(spec=ViTImageProcessor)

    #     train, val, test = get_and_prepare_data(
    #         "umaru97/flickr30k_train_val_test", tokenizer_mock, image_processor_mock
    #     )

    #     # Assertions
    #     mock_load_dataset.assert_any_call("flickr30k", split="train")
    #     mock_load_dataset.assert_any_call("flickr30k", split="val")
    #     mock_load_dataset.assert_any_call("flickr30k", split="test")
    #     self.assertEqual(len(mock_flickr_dataset.mock_calls), 3)
    #     self.assertIsNotNone(train)
    #     self.assertIsNotNone(val)
    #     self.assertIsNotNone(test)

    @patch("transformers.ViTImageProcessor.from_pretrained")
    @patch("transformers.GPT2TokenizerFast.from_pretrained")
    def test_load_preprocessors(self, mock_tokenizer, mock_image_processor):
        # Mock return values
        mock_tokenizer_instance = MagicMock(spec=GPT2TokenizerFast)
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_image_processor_instance = MagicMock(spec=ViTImageProcessor)
        mock_image_processor.return_value = mock_image_processor_instance

        image_processor, tokenizer = load_preprocessors(
            "vit_model_path", "gpt_model_path"
        )

        # Assertions
        mock_image_processor.assert_called_once_with("vit_model_path")
        mock_tokenizer.assert_called_once_with("gpt_model_path")
        self.assertIsNotNone(image_processor)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)

    @patch("transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained")
    def test_initialize_model(self, mock_model):
        # Mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Mock tokenizer attributes
        tokenizer_mock = MagicMock(spec=GPT2TokenizerFast)
        tokenizer_mock.bos_token_id = 50256
        tokenizer_mock.pad_token_id = 50256
        tokenizer_mock.eos_token_id = 50256

        # Correct decoder model identifier (use 'gpt2' instead of 'gpt_model')
        model = initialize_model(
            encoder="vit_model",
            decoder="gpt2",  # Corrected here
            max_length=50,
            early_stoping=True,
            no_repeat_ngram=2,
            num_beans=4,
        )

        # Assertions
        # Updated to match the new decoder model name
        mock_model.assert_called_once_with("vit_model", "gpt2")
        self.assertEqual(model.config.max_length, 50)
        self.assertTrue(model.config.early_stopping)
        self.assertEqual(model.config.no_repeat_ngram_size, 2)
        self.assertEqual(model.config.num_beams, 4)


if __name__ == "__main__":
    unittest.main()
