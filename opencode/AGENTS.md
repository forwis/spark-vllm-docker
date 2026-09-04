# OpenCode model-switch scripts

This directory contains executable scripts that replace the configured vLLM model in OpenCode. Each script is standalone and targets one model.

## Creating a model script

1. Copy the closest existing `*.sh` model script and name the new file after the lowercase model ID used by the local vLLM server.
2. Research the current model instead of copying another model's inference values:
   - Read the official Hugging Face model card, `config.json`, and `generation_config.json`.
   - Read the model's official vLLM recipe or serving documentation for chat-template and reasoning controls.
   - Query `https://openrouter.ai/api/v1/models` and match `hugging_face_id` exactly. Convert its per-token `pricing.prompt` and `pricing.completion` values to OpenCode's per-million-token `cost.input` and `cost.output` values by multiplying by 1,000,000.
   - Prefer the Hugging Face checkpoint's native context and output limits for local vLLM metadata. Do not substitute OpenRouter routing limits unless the script targets OpenRouter.
3. Configure only capabilities the sources establish: attachments, reasoning, temperature, tool calls, interleaved reasoning field, modalities, context/output limits, sampling parameters, and chat-template arguments.
4. Make the model's strongest documented reasoning profile the model default. Configure agents according to their workload:
   - `general`: strongest/deepest supported effort.
   - `scout`: middle supported effort.
   - `explore`: lowest supported effort, with thinking disabled only when the model explicitly supports non-thinking mode.
5. Preserve existing agent `description` and `mode`, remove stale agent fields, and add only the request settings needed by the target model. Ensure `general`, `scout`, and `explore` exist even when absent from the original configuration.

Do not assume that Qwen sampling penalties, GLM's `clear_thinking`, or DeepSeek's `thinking` flag apply to another model. Field names in `chat_template_kwargs` must match the serving implementation exactly. For `@ai-sdk/openai-compatible`, use camel-case `reasoningEffort` for AI SDK translation and retain any model-required snake-case chat-template argument.

## Automatic selection

`auto.sh` queries the vLLM `/models` endpoint using `VLLM_API_KEY` and the optional `VLLM_BASE_URL`, then considers only the first returned model ID. It searches this directory for that exact ID plus `.sh` and executes the file. Model script filenames must therefore exactly match the ID exposed through vLLM's `--served-model-name`; automatic selection does not lowercase IDs, remove organization prefixes, normalize aliases, or try later models.

Keep `auto.sh` fail-safe: an unavailable/malformed/empty response or a missing exact script must exit without running any model script or modifying the OpenCode configuration. Reject model IDs containing path separators so API data cannot select a script outside this directory.

## Safety invariants

- Resolve the config as `${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json` so tests can isolate it.
- Reject missing files and symbolic links before reading or writing.
- Parse JSONC comments and trailing commas without altering comment-like text inside strings.
- Preserve unrelated root configuration and vLLM connection settings; replace only `provider.vllm.models`, the root `model`, and agent request settings.
- Validate the complete in-memory structure before creating a backup.
- Create a timestamped `opencode.json.bak.*` backup, preserve permissions, and atomically replace the original from a temporary file in the same directory.
- Make repeated runs a no-op when the canonical output is already installed.
- Never run a model script against the user's live configuration during development or tests.

## Test workflow

Use test-driven development for every behavior change. Add or update a black-box `unittest` first, run it and confirm the expected failure, then implement the script.

Tests must set `XDG_CONFIG_HOME` to a `tempfile.TemporaryDirectory`, invoke the real shell script, and assert observable output and filesystem behavior. At minimum, cover exact model metadata and agent profiles, removal of old models and stale agent fields, preservation of unrelated settings, backup creation, idempotence, malformed input, missing config, and symlink rejection. Human documentation does not need a source-text test.

Before completion, run:

```bash
bash -n ./*.sh
python3 -m unittest -v
```

Also generate each model configuration under a temporary `XDG_CONFIG_HOME` and validate it with the installed parser:

```bash
XDG_CONFIG_HOME="$temporary_config_root" opencode --pure debug config
```

Do not modify the real `~/.config/opencode/opencode.json` as part of verification.
