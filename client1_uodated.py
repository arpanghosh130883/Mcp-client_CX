import asyncio
 
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
import json
import os
from pprint import pformat
 
load_dotenv()
 
SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "/opt/anaconda3/bin/python",
        "args": [
            "/Users/goutham/Desktop/FastMCPDemoServer/MathMCPServer.py"
        ],
    },
 
    "manim-server": {
        "transport": "stdio",
        "command": "/opt/anaconda3/bin/python",
        "args": [
            "/Users/goutham/manim-mcp-server/src/manim_server.py"
        ],
        "env": {
            "MANIM_EXECUTABLE": "/opt/anaconda3/bin/manim"
        }
    }
}
 
 
async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
 
    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool
 
    print("Available tools:", named_tools.keys())
 
    # Require OPENAI_API_KEY in env for ChatOpenAI; give a clear error if missing
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to your environment or .env file.")
        return
    llm = ChatOpenAI(model="gpt-5")
    llm_with_tools = llm.bind_tools(tools)
 
    prompt = "Draw a triangle rotating in place using the manim tool."
    response = await llm_with_tools.ainvoke(prompt)
 
    # Debug: print the raw LLM response and any tool calls it requested
    print("\n--- LLM initial response ---")
    try:
        print(pformat(response.__dict__))
    except Exception:
        print(repr(response))
 
    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", getattr(response, "content", ""))
        return
 
    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc.get("id")
 
        print(f"\nInvoking tool: {selected_tool} (id={selected_tool_id}) with args: {pformat(selected_tool_args)}")
        try:
            result = await named_tools[selected_tool].ainvoke(selected_tool_args)
            print(f"Tool {selected_tool} result: {pformat(result)}")
        except Exception as e:
            result = {"error": str(e)}
            print(f"Tool {selected_tool} raised error: {e}")
 
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
 
    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print("\n--- Final response ---")
    try:
        print(getattr(final_response, "content", ""))
    except Exception:
        print(repr(final_response))
 
 
if __name__ == '__main__':
    asyncio.run(main())
