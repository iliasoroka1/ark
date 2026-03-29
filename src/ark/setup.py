"""Interactive setup wizard for Ark.

Walks through embedding model selection, API keys, data directory,
dreamer config, and writes ~/.ark/config.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

console = Console()


def _select_menu(items: list[dict], label_key: str, detail_key: str, current_idx: int = 0) -> int:
    """Arrow-key navigable selection menu. Returns selected index."""
    import tty
    import termios

    def _read_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'up'
                    if ch3 == 'B': return 'down'
                return 'esc'
            if ch in ('\r', '\n'): return 'enter'
            if ch.isdigit(): return ch
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _draw(idx: int) -> None:
        for i, item in enumerate(items):
            if i == idx:
                console.print(f"  [bold cyan]> {i+1}. {item[label_key]}[/]  [dim]{item[detail_key]}[/]")
            else:
                console.print(f"    {i+1}. [dim]{item[label_key]}[/]  [dim]{item[detail_key]}[/]")

    idx = current_idx
    _draw(idx)

    while True:
        key = _read_key()
        if key == 'up':
            idx = (idx - 1) % len(items)
        elif key == 'down':
            idx = (idx + 1) % len(items)
        elif key == 'enter':
            # Redraw final state clean
            sys.stdout.write(f"\x1b[{len(items)}A\x1b[J")
            sys.stdout.flush()
            _draw(idx)
            return idx
        elif key.isdigit() and 1 <= int(key) <= len(items):
            idx = int(key) - 1
            sys.stdout.write(f"\x1b[{len(items)}A\x1b[J")
            sys.stdout.flush()
            _draw(idx)
            return idx
        else:
            continue

        # Move cursor up and clear, then redraw
        sys.stdout.write(f"\x1b[{len(items)}A\x1b[J")
        sys.stdout.flush()
        _draw(idx)

CONFIG_PATH = Path.home() / ".ark" / "config.json"

EMBEDDING_MODELS = [
    {
        "id": "pplx-embed",
        "name": "Perplexity pplx-embed-v1-0.6b",
        "type": "API (OpenRouter)",
        "dims": 1024,
        "env": {"OPENROUTER_EMBED_MODEL": "perplexity/pplx-embed-v1-0.6b"},
        "requires_key": True,
        "note": "Best quality. Requires OPENROUTER_API_KEY.",
    },
    {
        "id": "bge-large",
        "name": "BGE-large-en-v1.5",
        "type": "Local (fastembed)",
        "dims": 1024,
        "env": {"FASTEMBED_MODEL": "BAAI/bge-large-en-v1.5"},
        "requires_key": False,
        "note": "Good quality, runs locally. ~1.2GB download on first use.",
    },
    {
        "id": "bge-base",
        "name": "BGE-base-en-v1.5",
        "type": "Local (fastembed)",
        "dims": 768,
        "env": {"FASTEMBED_MODEL": "BAAI/bge-base-en-v1.5"},
        "requires_key": False,
        "note": "Balanced quality/speed. ~440MB download.",
    },
    {
        "id": "nomic",
        "name": "Nomic-embed-text-v1.5",
        "type": "Local (fastembed)",
        "dims": 768,
        "env": {},
        "requires_key": False,
        "note": "Default fallback. No config needed.",
    },
]

DREAMER_MODELS = [
    {
        "id": "gemini-flash",
        "name": "Gemini 3.1 Flash Lite",
        "model": "google/gemini-3.1-flash-lite-preview",
        "note": "Fast and cheap. Good for most use cases.",
    },
    {
        "id": "haiku",
        "name": "Claude Haiku 4.5",
        "model": "anthropic/claude-haiku-4-5-20251001",
        "note": "Higher quality reasoning. Slightly more expensive.",
    },
    {
        "id": "none",
        "name": "Disable dreamer",
        "model": "",
        "note": "No background memory consolidation.",
    },
]


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def _step_header(n: int, title: str) -> None:
    console.print()
    console.print(f"  [bold cyan]Step {n}[/] [dim]—[/] {title}")
    console.print()


def run_setup() -> None:
    config = load_config()

    console.print()
    console.print(Panel(
        "[bold]Ark Setup Wizard[/]\n\n"
        "Configure your embedding model, API keys, and data directory.\n"
        f"Config will be saved to [cyan]{CONFIG_PATH}[/]",
        border_style="blue",
        padding=(1, 2),
    ))

    # ── Step 1: Embedding model ──
    _step_header(1, "Embedding Model")

    console.print("  [dim]Use arrow keys to navigate, Enter to select (or type 1-4)[/]\n")

    current = config.get("embedding_model", "nomic")
    current_idx = next((i for i, m in enumerate(EMBEDDING_MODELS) if m["id"] == current), 0)
    choice = _select_menu(EMBEDDING_MODELS, "name", "note", current_idx)
    embed_model = EMBEDDING_MODELS[choice]
    config["embedding_model"] = embed_model["id"]
    config["embedding_dims"] = embed_model["dims"]
    config["embedding_env"] = embed_model["env"]
    console.print(f"  [green]Selected:[/] {embed_model['name']}")

    # ── Step 2: API Key ──
    needs_key = embed_model["requires_key"]
    current_key = config.get("openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", ""))

    if needs_key or Confirm.ask("\n  Set up OpenRouter API key? (needed for LLM expansion + dreamer)", default=bool(current_key)):
        _step_header(2, "OpenRouter API Key")
        masked = f"{current_key[:12]}...{current_key[-4:]}" if len(current_key) > 16 else ""
        if masked:
            console.print(f"  Current key: [dim]{masked}[/]")
        key = Prompt.ask("  Enter API key (or press Enter to keep current)", default=current_key)
        if key:
            config["openrouter_api_key"] = key
            console.print("  [green]Key saved[/]")
    else:
        _step_header(2, "OpenRouter API Key")
        console.print("  [dim]Skipped — local embedding only, no LLM expansion or dreamer[/]")

    # ── Step 3: Data directory ──
    _step_header(3, "Data Directory")
    current_home = config.get("ark_home", str(Path.home() / ".ark"))
    ark_home = Prompt.ask("  ARK_HOME", default=current_home)
    config["ark_home"] = ark_home
    Path(ark_home).mkdir(parents=True, exist_ok=True)
    console.print(f"  [green]Data directory:[/] {ark_home}")

    # ── Step 4: Dreamer ──
    _step_header(4, "Dreamer (Memory Consolidation)")
    console.print("  The dreamer runs an LLM to find knowledge updates,")
    console.print("  contradictions, and logical implications in your memories.\n")

    console.print("  [dim]Use arrow keys to navigate, Enter to select (or type 1-3)[/]\n")

    current_dreamer = config.get("dreamer_model_id", "gemini-flash")
    current_idx2 = next((i for i, m in enumerate(DREAMER_MODELS) if m["id"] == current_dreamer), 0)
    choice2 = _select_menu(DREAMER_MODELS, "name", "note", current_idx2)
    dreamer = DREAMER_MODELS[choice2]
    config["dreamer_model_id"] = dreamer["id"]
    config["dreamer_model"] = dreamer["model"]
    console.print(f"  [green]Selected:[/] {dreamer['name']}")

    # ── Step 5: Server port ──
    _step_header(5, "Server")
    port = IntPrompt.ask("  HTTP server port", default=int(config.get("port", 7070)))
    config["port"] = port

    # ── Save ──
    save_config(config)

    console.print()
    console.print(Panel(
        "[bold green]Setup complete![/]\n\n"
        f"Config saved to [cyan]{CONFIG_PATH}[/]\n\n"
        "Start ark:\n"
        f"  [bold]ark serve[/]         — start the HTTP server on port {port}\n"
        "  [bold]ark search[/] [dim]\"query\"[/] — search your knowledge\n"
        "  [bold]ark dream[/]          — run memory consolidation",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Generate shell env export ──
    env_lines = []
    if config.get("openrouter_api_key"):
        env_lines.append(f'export OPENROUTER_API_KEY="{config["openrouter_api_key"]}"')
    for k, v in config.get("embedding_env", {}).items():
        env_lines.append(f'export {k}="{v}"')
    if config.get("ark_home") and config["ark_home"] != str(Path.home() / ".ark"):
        env_lines.append(f'export ARK_HOME="{config["ark_home"]}"')
    if config.get("dreamer_model"):
        env_lines.append(f'export DREAMER_MODEL="{config["dreamer_model"]}"')

    if env_lines:
        console.print()
        console.print("  [dim]Add to your shell profile (.bashrc / .zshrc):[/]")
        console.print()
        for line in env_lines:
            console.print(f"    [cyan]{line}[/]")
        console.print()
