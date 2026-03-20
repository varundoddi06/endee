"""
agent.py
Agentic research loop using OpenAI function-calling.

The agent has access to two tools:
  1. search_memory   — searches Endee vector DB for relevant document chunks
  2. synthesize      — signals the agent is ready to produce a final answer

The loop runs until the model calls `synthesize` or hits MAX_STEPS.
"""

from __future__ import annotations
import json
import openai
from endee_client import EndeeVectorStore

MAX_STEPS = 6

# ── Tool schemas (OpenAI function-calling format) ────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the Endee vector database for document chunks relevant "
                "to a query. Returns the top matching passages and their sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the vector store.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize",
            "description": (
                "Call this when you have gathered enough information from memory "
                "and are ready to produce the final answer for the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete, well-structured final answer.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are ResearchMind, an expert agentic research assistant.
You have access to a vector memory (Endee vector database) that contains documents 
the user has uploaded. Your job is to answer research questions by:

1. Deciding what to search for in memory.
2. Calling search_memory one or more times to retrieve relevant chunks.
3. Reasoning over the retrieved passages.
4. Calling synthesize with a comprehensive, well-structured answer.

Guidelines:
- Always search memory before answering. Do at least one search.
- If initial results are insufficient, search again with a refined query.
- Cite sources (filename and chunk) when possible.
- Be analytical, not just extractive. Summarize, compare, and interpret.
- Structure the final answer with headers if it covers multiple points.
"""


class ResearchAgent:
    def __init__(self, vector_store: EndeeVectorStore, openai_key: str):
        self.vs = vector_store
        self.oai = openai.OpenAI(api_key=openai_key)

    def run(self, user_query: str, log_callback=None) -> str:
        """
        Execute the agentic loop and return the final answer.
        log_callback(dict) is called for each observable step.
        """
        def log(type_: str, label: str, content: str):
            if log_callback:
                log_callback({"type": type_, "label": label, "content": content})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        log("thought", "Received query", user_query)

        for step in range(MAX_STEPS):
            response = self.oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            # ── No tool call → model gave a text answer directly ─────────────
            if not msg.tool_calls:
                log("answer", "Final Answer", msg.content or "")
                return msg.content or "I could not find an answer."

            # ── Process each tool call ────────────────────────────────────────
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)

                if fn_name == "search_memory":
                    query = args["query"]
                    top_k = args.get("top_k", 5)

                    log("memory", f"Searching Endee (top {top_k})", f'"{query}"')

                    hits = self.vs.search(query=query, openai_key=self.oai.api_key, top_k=top_k)

                    # Format results for the model
                    if hits:
                        result_text = "\n\n".join(
                            f"[{h['source']} | chunk {h['chunk']} | score {h['score']}]\n{h['text']}"
                            for h in hits
                        )
                        log("tool", "Memory results", f"Retrieved {len(hits)} chunks from Endee")
                    else:
                        result_text = "No results found for this query."
                        log("tool", "Memory results", "No results found.")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                elif fn_name == "synthesize":
                    answer = args.get("answer", "")
                    log("answer", "Synthesized Answer", answer[:200] + ("..." if len(answer) > 200 else ""))
                    return answer

        # ── Fallback if MAX_STEPS hit ─────────────────────────────────────────
        log("answer", "Max steps reached", "Returning best available answer.")
        last_text = next(
            (m.content for m in reversed(messages) if hasattr(m, "content") and m.content),
            "I was unable to complete the research within the step limit."
        )
        return last_text
