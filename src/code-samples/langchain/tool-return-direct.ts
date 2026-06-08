// :snippet-start: tool-return-direct-js
import { ChatOpenAI } from "@langchain/openai";
import { ToolMessage } from "@langchain/core/messages";
import { createAgent, tool } from "langchain";
import { MemorySaver } from "@langchain/langgraph";
import * as z from "zod";

const fetchOrderStatus = tool(
  ({ order_id }) => {
    return `Order ${order_id} is shipped and will arrive in 2 days.`;
  },
  {
    name: "fetch_order_status",
    description: "Fetch the current status of a customer order.",
    schema: z.object({ order_id: z.string() }),
    returnDirect: true,
  },
);

const agent = createAgent({
  model: new ChatOpenAI({ model: "gpt-4o-mini" }),
  tools: [fetchOrderStatus],
  checkpointer: new MemorySaver(),
});

const config = { configurable: { thread_id: crypto.randomUUID() } };

let result = await agent.invoke(
  {
    messages: [
      { role: "user", content: "What is the status of order #12345?" },
    ],
  },
  config,
);
// The agent returns the tool output directly without another LLM call:
// "Order 12345 is shipped and will arrive in 2 days."

result = await agent.invoke(
  { messages: [{ role: "user", content: "When will it arrive?" }] },
  config,
);
// :snippet-end:

// :remove-start:
async function main() {
  if (!fetchOrderStatus.returnDirect) {
    throw new Error("Expected fetchOrderStatus.returnDirect to be true");
  }

  const toolMsgs = result.messages.filter((m) => m instanceof ToolMessage);
  if (toolMsgs.length === 0) {
    throw new Error(
      "expected ToolMessage from return_direct turn in thread history",
    );
  }
  if (
    !toolMsgs.some(
      (m) =>
        String(m.content).includes("shipped") &&
        String(m.content).includes("12345"),
    )
  ) {
    throw new Error("expected direct tool output for order 12345 in history");
  }

  const humanCount = result.messages.filter((m) => m.type === "human").length;
  if (humanCount < 2) {
    throw new Error(`expected >=2 human turns, got ${humanCount}`);
  }

  const last = result.messages.at(-1);
  const content =
    typeof last?.content === "string"
      ? last.content
      : JSON.stringify(last?.content);

  if (!content) {
    throw new Error("expected non-empty follow-up assistant reply");
  }

  console.log(
    "✓ returnDirect tool sample completed with follow-up on same thread",
  );
}

main();
// :remove-end:
