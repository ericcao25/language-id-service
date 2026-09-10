import argparse
import asyncio
import base64
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main(url: str, audio_path: str | None):
    print(f"Connecting to {url} ...")
    async with streamablehttp_client(url) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected. Listing tools...")

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"Available tools: {tool_names}")

            assert "list_supported_regions" in tool_names, "expected tool not found"
            assert "predict_language" in tool_names, "expected tool not found"

            print("\nCalling list_supported_regions")
            regions_result = await session.call_tool("list_supported_regions", {})
            print(json.dumps(_extract(regions_result), indent=2))

            if audio_path:
                print(f"\nCalling predict_language on {audio_path}")
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("ascii")

                predict_result = await session.call_tool(
                    "predict_language",
                    {"audio_base64": audio_b64, "top_k": 3},
                )
                print(json.dumps(_extract(predict_result), indent=2))
            else:
                print("\n(no --audio given, skipping predict_language call)")


def _extract(call_tool_result):
    """CallToolResult content is a list of content blocks; pull out text/JSON."""
    out = []
    for block in call_tool_result.content:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                out.append(text)
        else:
            out.append(str(block))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--audio", default=None)
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url, args.audio))
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
