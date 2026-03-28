"""Seed 500 tech news articles as hard distractors."""
import asyncio
import json

async def main():
    from ark.local import call_tool
    with open('/Users/iliasoroka/ark/tech_noise.json') as f:
        samples = json.load(f)
    total = len(samples)
    for i, text in enumerate(samples):
        await call_tool('ingest', {'content': text})
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f'  [{i+1}/{total}]')
    print(f'Done: {total} tech noise articles seeded')

if __name__ == '__main__':
    asyncio.run(main())
