import argparse
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Reproducible KION recommendation baseline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("command", choices=["config", "download"], help="Pipeline stage")
    parser.add_argument("--archive", help="Existing pinned data_original.zip")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    if args.command == "config":
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
    elif args.command == "download":
        from .data import audit_archive
        import json
        print(json.dumps(audit_archive(cfg, args.archive)[3], ensure_ascii=False, indent=2))
