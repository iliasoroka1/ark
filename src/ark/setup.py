"""Interactive setup wizard for Ark.

Walks through embedding model selection, API keys, data directory,
dreamer config, and writes ~/.ark/config.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

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

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2), expand=True)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Model", width=26, no_wrap=True)
    table.add_column("Type", width=16, no_wrap=True)
    table.add_column("Dims", width=5)
    table.add_column("Notes", style="dim", ratio=1)

    for i, m in enumerate(EMBEDDING_MODELS, 1):
        table.add_row(str(i), m["name"], m["type"], str(m["dims"]), m["note"])

    console.print(table)
    console.print()

    current = config.get("embedding_model", "nomic")
    choice = IntPrompt.ask(
        "  Choose embedding model",
        default=next((i for i, m in enumerate(EMBEDDING_MODELS, 1) if m["id"] == current), 1),
        choices=[str(i) for i in range(1, len(EMBEDDING_MODELS) + 1)],
    )
    embed_model = EMBEDDING_MODELS[choice - 1]
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
        key = Prompt.ask("  Enter API key (or press Enter to keep current)", default=current_key, password=True)
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

    table2 = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table2.add_column("#", style="cyan", width=3)
    table2.add_column("Model", width=24)
    table2.add_column("Notes", style="dim")

    for i, m in enumerate(DREAMER_MODELS, 1):
        table2.add_row(str(i), m["name"], m["note"])

    console.print(table2)
    console.print()

    current_dreamer = config.get("dreamer_model_id", "gemini-flash")
    choice2 = IntPrompt.ask(
        "  Choose dreamer model",
        default=next((i for i, m in enumerate(DREAMER_MODELS, 1) if m["id"] == current_dreamer), 1),
        choices=[str(i) for i in range(1, len(DREAMER_MODELS) + 1)],
    )
    dreamer = DREAMER_MODELS[choice2 - 1]
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
