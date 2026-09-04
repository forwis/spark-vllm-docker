#!/usr/bin/env bash
set -euo pipefail

config_home="${XDG_CONFIG_HOME:-${HOME:?HOME is not set}/.config}"
config_path="${config_home}/opencode/opencode.json"

if [[ -L "$config_path" ]]; then
  printf 'Error: refusing to replace symbolic link: %s\n' "$config_path" >&2
  exit 1
fi

if [[ ! -f "$config_path" ]]; then
  printf 'Error: OpenCode config not found: %s\n' "$config_path" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'Error: python3 is required to update %s\n' "$config_path" >&2
  exit 1
fi

python3 - "$config_path" <<'PY'
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def strip_jsonc(source):
    result = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            result.extend("  ")
            index += 2
            while index < len(source) and source[index] not in "\r\n":
                result.append(" ")
                index += 1
            continue
        if char == "/" and following == "*":
            result.extend("  ")
            index += 2
            while index < len(source):
                if index + 1 < len(source) and source[index] == "*" and source[index + 1] == "/":
                    result.extend("  ")
                    index += 2
                    break
                result.append(source[index] if source[index] in "\r\n" else " ")
                index += 1
            else:
                raise ValueError("unterminated block comment")
            continue
        result.append(char)
        index += 1
    if in_string:
        raise ValueError("unterminated string")
    return remove_trailing_commas("".join(result))


def remove_trailing_commas(source):
    result = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(source) and source[lookahead].isspace():
                lookahead += 1
            if lookahead < len(source) and source[lookahead] in "}]":
                index += 1
                continue
        result.append(char)
        index += 1
    return "".join(result)


def sampling_options(temperature, top_p, reasoning_effort, thinking):
    return {
        "temperature": temperature,
        "top_p": top_p,
        "reasoningEffort": reasoning_effort,
        "chat_template_kwargs": {
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
        },
    }


config_path = Path(sys.argv[1])
original_text = config_path.read_text(encoding="utf-8")
try:
    config = json.loads(strip_jsonc(original_text))
except (json.JSONDecodeError, ValueError) as error:
    print(f"Error: invalid JSONC in {config_path}: {error}", file=sys.stderr)
    raise SystemExit(1)

if not isinstance(config, dict):
    print(f"Error: invalid JSONC in {config_path}: root must be an object", file=sys.stderr)
    raise SystemExit(1)
provider = config.get("provider")
if not isinstance(provider, dict) or not isinstance(provider.get("vllm"), dict):
    print(f"Error: provider.vllm must be an object in {config_path}", file=sys.stderr)
    raise SystemExit(1)

provider["vllm"]["models"] = {
    "deepseek-v4-flash-vision-exp": {
        "name": "DeepSeek-V4-Flash-Vision-Exp",
        "cost": {"input": 0.22, "output": 0.66},
        "attachment": True,
        "reasoning": True,
        "temperature": True,
        "tool_call": True,
        "interleaved": "reasoning_content",
        "limit": {"context": 1048576, "output": 393216},
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "options": sampling_options(1.0, 0.95, "max", True),
    }
}
config["model"] = "vllm/deepseek-v4-flash-vision-exp"

profiles = {
    "general": (1.0, 0.95, "max", True),
    "scout": (1.0, 0.95, "high", True),
    "explore": (1.0, 1.0, "low", False),
}
agent_defaults = {
    "general": {
        "description": "A general-purpose agent for researching complex questions and executing multi-step tasks. Has full tool access (except todo), so it can make file changes when needed. Use this to run multiple units of work in parallel.",
        "mode": "subagent",
    },
    "scout": {
        "description": "A read-only agent for external docs and dependency research. Use this when you need to clone a dependency repository into OpenCode's managed cache, inspect library source, or cross-reference local code against upstream implementations without modifying your workspace.",
        "mode": "subagent",
    },
    "explore": {
        "description": "A fast, read-only agent for exploring codebases. Cannot modify files. Use this when you need to quickly find files by patterns, search code for keywords, or answer questions about the codebase.",
        "mode": "subagent",
    },
}
agents = config.get("agent")
if agents is None:
    agents = {}
    config["agent"] = agents
elif not isinstance(agents, dict):
    print(f"Error: agent must be an object in {config_path}", file=sys.stderr)
    raise SystemExit(1)
for name, defaults in agent_defaults.items():
    agents.setdefault(name, defaults.copy())
for name, agent in agents.items():
    if not isinstance(agent, dict):
        print(f"Error: agent.{name} must be an object in {config_path}", file=sys.stderr)
        raise SystemExit(1)
    profile = profiles.get(name, profiles["general"])
    cleaned = {
        key: agent.get(key, agent_defaults.get(name, {}).get(key))
        for key in ("description", "mode")
        if key in agent or key in agent_defaults.get(name, {})
    }
    cleaned.update(
        {
            "temperature": profile[0],
            "top_p": profile[1],
            "options": sampling_options(*profile),
        }
    )
    agents[name] = cleaned

updated_text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
if original_text == updated_text:
    print(f"Already configured: {config_path}")
    raise SystemExit(0)

backup_suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
backup_path = config_path.with_name(f"{config_path.name}.bak.{backup_suffix}")
shutil.copy2(config_path, backup_path)
mode = config_path.stat().st_mode
temporary_path = None
try:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{config_path.name}.", dir=config_path.parent)
    temporary_path = Path(temporary_name)
    os.fchmod(descriptor, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(updated_text)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, config_path)
except BaseException:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
    raise

print(f"Updated: {config_path}")
print(f"Backup:  {backup_path}")
PY
