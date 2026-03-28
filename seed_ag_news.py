"""Seed 500 AG News articles as scale noise for harder benchmark."""
import asyncio
import json

async def main():
    from ark.local import call_tool

    with open('/Users/iliasoroka/ark/ag_news_noise.json') as f:
        samples = json.load(f)

    total = len(samples)
    for i, text in enumerate(samples):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f'  [{i+1}/{total}] ingested')

    print(f'Done: {total} AG News memories seeded')

if __name__ == '__main__':
    asyncio.run(main())
