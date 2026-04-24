"""TOML-backed training configuration example.

The TOML file configures model and trainer setup only. Real datasets, splits,
and loaders remain caller-provided through the existing `Trainer` workflow.
"""

from pathlib import Path
import shutil

import torch

from phylognn.training import create_trainer_from_config, load_training_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "toml_training_config.toml"
OUTPUT_DIR = ROOT / "example_outputs" / "toml_training_config"


def main() -> None:
    torch.manual_seed(7)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    setup = load_training_config(
        CONFIG_PATH,
        training_overrides={"save_dir": str(OUTPUT_DIR), "verbose": False},
    )
    trainer = create_trainer_from_config(
        CONFIG_PATH,
        training_overrides={"save_dir": str(OUTPUT_DIR), "verbose": False},
    )

    print("TOML training config summary")
    print(f"configured model: {setup.model.__class__.__name__}")
    print(f"input_dim: {setup.model.input_dim}")
    print(f"output_dim: {setup.model.output_dim}")
    print(f"epochs: {setup.training_config.epochs}")
    print(f"batch_size: {setup.training_config.batch_size}")
    print(f"trainer device: {trainer.config.device}")
    print(f"metrics: {', '.join(setup.metrics)}")
    print(f"output_dir: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
