"""Public research CLI. Import numeric libraries only inside the capped worker."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import ResearchConfig

COMMANDS = ("prepare", "search", "validate", "freeze", "test", "verify", "reproduce")
STAGES = ("prepare", "search", "validate", "freeze", "test", "verify")


def execute(command, research):
    from .runtime import configure
    configure(research.threads)
    if command == "prepare":
        from .baseline import Baseline
        return Baseline(research).prepare()
    if command == "search":
        from .search import run_search
        return run_search(research)
    if command == "validate":
        from .experiment import run_validation
        return run_validation(research)
    from .final import freeze, run_final, verify
    return {"freeze": freeze, "test": run_final, "verify": verify}[command](research)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible temporal automatic-feature research")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="autofeatures.toml", help="Research TOML configuration")
    parser.add_argument("--name", help="Separate immutable research run name")
    parser.add_argument("--through", choices=STAGES, default="verify", help="Last stage of a fresh reproduction")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    research = ResearchConfig.load(config_path)
    if args.name:
        research = replace(research, run_name=args.name)
    if args.worker:
        if args.command == "reproduce":
            parser.error("A worker executes one stage only")
        result = execute(args.command, research)
        print(json.dumps({"stage": args.command, "run": research.run_name,
                          "test_evaluated": result.get("test_evaluated", False)}, ensure_ascii=False), flush=True)
        return 0
    from .runtime import execution_lock, guarded_stage
    if args.command == "reproduce":
        if not args.name:
            parser.error("Use reproduce --name NEW_NAME to preserve previous results")
        if research.artifacts.exists() or research.results.exists():
            parser.error("Reproduction requires a new, unused name")
        commands = STAGES[:STAGES.index(args.through) + 1]
    else:
        commands = (args.command,)
    with execution_lock(research.root):
        for command in commands:
            guarded_stage(command, config_path, research)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
