# Clarus Agent (VS Code)

A minimal Claude Code / Codex-style coding agent that lives in a VS Code
sidebar panel — but every side-effecting action (`write_file`, `run_command`)
is intercepted by [ClarusGuard](../README.md) before it touches disk or a
shell:

- **capability check (I1)** — a write/run action only executes if the user's
  own message actually asked for it; content injected via tool output can
  never mint that authority.
- **taint check (I2)** — even an authorised call is refused if a critical
  argument (e.g. a file path) was lifted from untrusted data.
- **human approval** — every side-effecting call also pauses for an explicit
  Approve/Deny prompt before it runs.

## Architecture

```
VS Code webview (media/main.js)
  <-> chatViewProvider.ts        (postMessage <-> AgentSession)
        <-> providers.ts         (Anthropic Messages API | OpenAI Chat Completions,
                                   both via native fetch — pick with clarusAgent.provider)
        <-> agentLoop.ts         (tool-use loop, resolves paths, requests approval)
              <-> guardBridge.ts (JSON-lines child_process)
                    <-> server/vscode_bridge.py  (real read_file/write_file/run_command,
                                                    gated by ClarusGuard)
```

The extension never touches the filesystem or a shell directly — every tool
call is a JSON line sent to the Python bridge, which is the only process that
executes anything, and only after `ClarusGuard.call()` clears both gates.

## Setup

```bash
cd vscode-extension
npm install
npm run compile
```

Press F5 in VS Code (with this folder open) to launch an Extension
Development Host, or package with `vsce package`.

Configure in settings (`clarusAgent.*`):

- `provider`: `"anthropic"` or `"openai"`
- `anthropicApiKey` / `openaiApiKey` (or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`)
- `pythonPath`: interpreter with the `clarus-agent-guard` repo importable
  (defaults to `python`; run `pip install -r ../requirements.txt` first)
- `guardRepoPath`: defaults to the `clarus-agent-guard` folder this extension
  ships alongside

## Status

Working prototype: chat panel, real tool-use loop against Anthropic or
OpenAI, real file/shell tools gated by ClarusGuard with modal approval. Not
yet done: streaming responses, diff-preview UI for `write_file`, multi-turn
session persistence, cancel-in-flight.
