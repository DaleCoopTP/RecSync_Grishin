import argparse
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Reproducible KION recommendation baseline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("command", choices=["config"], help="Pipeline stage")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    if args.command == "config":
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
