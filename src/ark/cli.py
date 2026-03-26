import asyncio
import json

import click

from .config import post


@click.group()
def ark():
    """ark — CLI for tinyclaw agent infrastructure."""
    pass


@ark.command()
def ping():
    """Check if the tinyclaw server is reachable."""

    async def _ping():
        try:
            result = await post("/health", {})
            print(json.dumps(result, indent=2))
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(_ping())


def main():
    # Import and register subcommand groups
    from .history import history
    ark.add_command(history)

    try:
        from .memory import memory
        ark.add_command(memory)
    except ImportError:
        pass

    try:
        from .rag import rag
        ark.add_command(rag)
    except ImportError:
        pass

    ark()


if __name__ == "__main__":
    main()
