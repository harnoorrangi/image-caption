from dataclasses import dataclass


@dataclass
class Model_Params:
    encoder_model: str
    decoder_model: str
    trained_model: str


@dataclass
class Dataset_Params:
    dataset: str


@dataclass
class Train_Params:
    train_batch_size: int
    val_batch_size: int
    test_batch_size: int
    val_epochs: int
    learning_rate: float
    epochs: int
    generation_max_length: int
    early_stoping_enabled: bool
    no_repeat_ngram_size: int
    num_beans: int
    hugging_face_model_id: str


@dataclass
class TrainingConfig:
    model_params: Model_Params
    dataset_params: Dataset_Params
    train_params: Train_Params
