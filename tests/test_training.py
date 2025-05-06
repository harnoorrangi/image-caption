import pytest
import torch

from image_caption.scripts.train_model import (
    get_and_prepare_data,
    initialize_model,
    load_preprocessors,
)


class DummyTokenizer:
    def __init__(self):
        self.eos_token = "<EOS>"
        self.pad_token = None
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.eos_token_id = 3


class DummyImageProcessor:
    pass


class DummyModel:
    def __init__(self):
        # mimic HuggingFace config object
        class C:
            pass

        self.config = C()


@pytest.fixture(autouse=True)
def patch_hf_and_datasets(monkeypatch):
    # patch VisionEncoderDecoderModel
    import transformers

    monkeypatch.setattr(
        transformers.VisionEncoderDecoderModel, "from_encoder_decoder_pretrained", lambda enc, dec: DummyModel()
    )

    # patch GPT2TokenizerFast.from_pretrained
    from transformers import GPT2TokenizerFast

    monkeypatch.setattr(GPT2TokenizerFast, "from_pretrained", lambda model_name: DummyTokenizer())

    # patch ViTImageProcessor.from_pretrained
    from transformers import ViTImageProcessor

    monkeypatch.setattr(ViTImageProcessor, "from_pretrained", lambda model_name: DummyImageProcessor())

    # patch load_dataset
    import image_caption.scripts.train_model as M

    monkeypatch.setattr(M, "load_dataset", lambda name, split: f"{name}-{split}")

    # patch Flickr30kDataset so it just echoes its inputs
    monkeypatch.setattr(M, "Flickr30kDataset", lambda data, tok, proc: (data, tok, proc))

    yield  # run tests with these patches in place


def test_load_preprocessors_returns_correct_types_and_pad_token():
    img_proc, tok = load_preprocessors("any-vit", "any-gpt2")
    assert isinstance(img_proc, DummyImageProcessor)
    assert isinstance(tok, DummyTokenizer)
    # because pad_token was set to eos_token
    assert tok.pad_token == "<EOS>"


def test_initialize_model_sets_config_fields():
    # pass dummy strings; our patch will return DummyModel + DummyTokenizer
    model = initialize_model(
        encoder="enc-model",
        decoder="dec-model",
        max_length=123,
        early_stopping=True,
        no_repeat_ngram=4,
        num_beams=7,
    )
    cfg = model.config
    # verify that every field made it into config
    assert cfg.max_length == 123
    assert cfg.early_stopping is True
    assert cfg.no_repeat_ngram_size == 4
    assert cfg.num_beams == 7
    # also check that start/pad/eos token ids came from DummyTokenizer
    assert cfg.decoder_start_token_id == 1
    assert cfg.pad_token_id == 2
    assert cfg.eos_token_id == 3


def test_get_and_prepare_data_returns_three_splits():
    dummy_tok = object()
    dummy_img = object()

    train_ds, val_ds, test_ds = get_and_prepare_data("flickr_dataset", dummy_tok, dummy_img)

    # since we patched load_dataset to return "name-split"
    assert train_ds == ("flickr_dataset-train", dummy_tok, dummy_img)
    assert val_ds == ("flickr_dataset-val", dummy_tok, dummy_img)
    assert test_ds == ("flickr_dataset-test", dummy_tok, dummy_img)
