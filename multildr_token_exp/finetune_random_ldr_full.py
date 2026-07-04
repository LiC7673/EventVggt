"""Strategy A: historical Full model with random exposure sampling only."""

from pathlib import Path
import sys

import hydra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multildr_token_exp.common import launch  # noqa: E402


@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="finetune_event.yaml")
def run(cfg):
    launch(cfg, "random_ldr_full")


if __name__ == "__main__":
    run()
