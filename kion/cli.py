import argparse
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Reproducible KION recommendation baseline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("command", choices=["config", "download", "prepare", "train", "evaluate", "recommend", "run"], help="Pipeline stage")
    parser.add_argument("--archive", help="Existing pinned data_original.zip")
    parser.add_argument("--smoke", action="store_true", help="1000 users, short validation-only technical run")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    if args.smoke:
        from .pipeline import smoke_config
        cfg = smoke_config(cfg)
    if args.command == "config":
        import json
        print(json.dumps(cfg.to_dict(), indent=2))
    elif args.command == "download":
        from .data import audit_archive
        import json
        print(json.dumps(audit_archive(cfg, args.archive)[3], ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        from .protocol import prepare
        import json
        print(json.dumps(prepare(cfg, args.archive), ensure_ascii=False, indent=2))
    elif args.command in {"train", "evaluate", "run"}:
        from .pipeline import run_validation
        import json
        result = run_validation(cfg, fit=args.command != "evaluate")
        print(json.dumps({"run_id": result["run_id"], "selected": result["selected"], "test_evaluated": False}, indent=2))
    elif args.command == "recommend":
        if args.user_id is None:
            parser.error("recommend requires --user-id")
        from .pipeline import recommend
        import json
        print(json.dumps(recommend(cfg, args.user_id, args.k), ensure_ascii=False, indent=2))
