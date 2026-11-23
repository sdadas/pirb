"""Resolve a Karpo checkpoint and start the native server on one GPU."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="MachinePulseAI/karpo-decision-31B",
                        help="Hugging Face repo ID or mounted checkpoint directory")
    parser.add_argument("--revision", help="Hugging Face checkpoint revision")
    parser.epilog = "Other options (e.g. --port, --name, --max-length, --fp8) go to the Karpo server."
    args, server_args = parser.parse_known_args()
    if any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in server_args):
        parser.error("use --model to select a local checkpoint or Hub repository")

    path = Path(args.model)
    if path.is_dir():
        if args.revision:
            parser.error("--revision requires a Hugging Face repo ID")
        checkpoint = str(path.resolve())
    else:
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(args.model, revision=args.revision)

    command = ["torchrun", "--standalone", "--nproc_per_node", "1",
               "-m", "karpo_decision.serve.server", "--checkpoint", checkpoint,
               "--host", "0.0.0.0", *server_args]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
