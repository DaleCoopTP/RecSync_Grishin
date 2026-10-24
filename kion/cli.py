import argparse
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="Reproducible KION recommendation baseline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("command", choices=["config", "download", "prepare", "train", "evaluate", "recommend", "run", "lock-test", "test", "verify-test"], help="Pipeline stage")
    parser.add_argument("--archive", help="Existing pinned data_original.zip")
    parser.add_argument("--reuse-prefix-from", help="Previous validation artifact directory; verify identical snapshots before reusing a model prefix")
    parser.add_argument("--smoke", action="store_true", help="1000 users, short validation-only technical run")
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--stage", choices=["auto", "validation", "test"], default="auto", help="Saved checkpoint for recommendations")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    # Set native-library limits before importing numpy, scipy or CatBoost.
    import os
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"]:
        os.environ[key] = str(cfg.threads)
    if args.reuse_prefix_from and args.command not in {"train", "run"}:
        parser.error("--reuse-prefix-from is supported only for train/run")
    if args.smoke and args.command in {"lock-test", "test", "verify-test"}:
        parser.error("Final evaluation uses the locked cohort; --smoke is only for validation")
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
        result = run_validation(cfg, fit=args.command != "evaluate", prefix_source=args.reuse_prefix_from)
        print(json.dumps({"run_id": result["run_id"], "selected": result["selected"], "test_evaluated": False}, indent=2))
    elif args.command in {"lock-test", "test", "verify-test"}:
        from .final import lock_test, run_final, verify_final
        import json
        if args.command == "lock-test":
            result = lock_test(cfg)
        elif args.command == "test":
            result = run_final(cfg)
            result = {"artifact_directory": result["artifact_directory"], "metrics": result["metrics"], "test_evaluated": True}
        else:
            result = verify_final(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        if args.user_id is None:
            parser.error("recommend requires --user-id")
        from .inference import recommend_saved
        import json
        print(json.dumps(recommend_saved(cfg, args.user_id, args.k, args.stage), ensure_ascii=False, indent=2))
