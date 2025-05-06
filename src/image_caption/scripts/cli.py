# src/image_caption/scripts/cli.py
import typer

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


if __name__ == "__main__":
    app()
