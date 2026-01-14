from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_config = root / "opencode.jsonc"
    config_default = default_config if default_config.exists() else None

    parser = argparse.ArgumentParser(
        description="Run an opencode prompt via subprocess.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        type=Path,
        help="Path to a prompt file to send to opencode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_default,
        help="Path to opencode.jsonc (defaults to project root if found).",
    )
    parser.add_argument(
        "--opencode-bin",
        default="opencode",
        help="Executable name or full path for opencode.",
    )
    parser.add_argument(
        "--mode",
        choices=("stdin", "arg"),
        default="arg",
        help="How to pass the prompt to opencode (stdin or prompt flag).",
    )
    parser.add_argument(
        "--prompt-flag",
        default="-f",
        help="Flag to use when mode=arg (use -f/--file for opencode).",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=root,
        help="Working directory for the opencode process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without running.",
    )
    return parser.parse_args()


def build_command(
    opencode_bin: str,
    config_path: Optional[Path],
    prompt_path: Path,
    mode: str,
    prompt_flag: str,
) -> tuple[list[str], Optional[str]]:
    cmd = [opencode_bin]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    cmd.append("run")

    if mode == "arg":
        # opencode는 --file/-f로 파일 첨부를 받음
        cmd.extend([prompt_flag, str(prompt_path)])
        # 메시지가 비면 아무 것도 안 할 수 있어서 짧은 지시를 같이 보냄
        cmd.append("Please read and execute the instructions in the attached file.")
        return cmd, None

    prompt_text = prompt_path.read_text(encoding="utf-8")
    return cmd, prompt_text


def main() -> int:
    args = parse_args()
    prompt_path = args.prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    cmd, stdin_text = build_command(
        args.opencode_bin,
        args.config,
        prompt_path,
        args.mode,
        args.prompt_flag,
    )

    print("Command:", " ".join(cmd))
    print("Working dir:", args.cwd)

    if args.dry_run:
        return 0

    result = subprocess.run(
        cmd,
        input=stdin_text,
        text=True,
        encoding="utf-8",
        cwd=str(args.cwd),
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
