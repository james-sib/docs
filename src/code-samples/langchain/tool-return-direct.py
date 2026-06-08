# :snippet-start: tool-return-direct-py
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    # In production, query your order management system here
    return f"Order {order_id} is shipped and will arrive in 2 days."


agent = create_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[fetch_order_status],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of order #12345?"}]},
    config=config,
)

# The agent returns the tool output directly without another LLM call:
# "Order 12345 is shipped and will arrive in 2 days."

result = agent.invoke(
    {"messages": [{"role": "user", "content": "When will it arrive?"}]},
    config=config,
)
# :snippet-end:
# :remove-start:
if __name__ == "__main__":
    from langchain.messages import ToolMessage

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected ToolMessage from return_direct turn in thread history"
    assert any(
        "shipped" in str(m.content) and "12345" in str(m.content) for m in tool_msgs
    )

    human_msgs = [m for m in result["messages"] if m.type == "human"]
    assert len(human_msgs) >= 2

    last = result["messages"][-1]
    text = (
        last.content_blocks[0]["text"]
        if getattr(last, "content_blocks", None)
        else str(last.content)
    )

    assert text, "expected non-empty follow-up assistant reply"
    print("✓ returnDirect tool sample completed with follow-up on same thread")
# :remove-end:
