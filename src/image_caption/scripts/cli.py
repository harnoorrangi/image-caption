from typing import List

import typer

from image_caption.scripts.predict_model import load_model, predict_caption
from image_caption.scripts.train_model import train_main
from image_caption.scripts.utils import load_hydra_config

app = typer.Typer()


@app.command()
def train(config_name: str = typer.Option(..., "--config-name", "-c", help="Hydra config name (no .yaml)")):
    """
    Train the model using the specified Hydra configuration.
    """
    cfg = load_hydra_config(config_name)
    train_main(cfg)


@app.command()
def print_hello():
    """
    Print hello world.
    """
    print("Hello, world!")


@app.command()
def predict(
    model_name: str = typer.Option(..., "--model-name", "-m", help="Hugging Face model name or path"),
    image_paths: List[str] = typer.Option(..., "--image-paths", "-i", help="Paths to the images"),
):
    """
    Predict captions for the specified image(s) using the given model.
    """
    # Load model and tools
    model, image_processor, tokenizer, device = load_model(model_name)

    # Loop over each image and generate caption
    for image_path in image_paths:
        caption = predict_caption(model, image_processor, tokenizer, device, image_path)
        typer.echo(f"Image: {image_path}")
        typer.echo(f"Caption: {caption}\n")


if __name__ == "__main__":
    app()
