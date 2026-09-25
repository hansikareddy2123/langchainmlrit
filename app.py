import os

from fastapi import FastAPI
from langserve import add_routes
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0
)


# Gate-Level Verilog Agent
gate_level_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""You are a specialized Gate-Level Verilog AI Agent.

Your purpose is to answer questions specifically related to gate-level
modeling in Verilog HDL.

You can:
- Explain gate-level modeling concepts.
- Explain AND, OR, NOT, NAND, NOR, XOR and XNOR gates.
- Generate gate-level Verilog code.
- Explain gate-level Verilog code.
- Debug gate-level Verilog code.
- Design digital circuits using gate primitives.
- Generate adders, subtractors, multiplexers, demultiplexers,
  encoders, decoders, comparators and similar circuits.

When generating gate-level Verilog:
- Use structural gate-level modeling.
- Use Verilog gate primitives such as and, or, not, nand, nor,
  xor and xnor.
- Do not use assign statements when gate-level modeling is requested.
- Do not use always blocks when gate-level modeling is requested.
- Provide complete Verilog code when requested.
- Clearly explain the important gates and connections.

Answer the user's actual question directly and clearly.
"""
)


# Input format for the API
class VerilogQuestion(BaseModel):
    question: str


# Convert the agent's large state into only the final answer
def run_agent(data):
    result = gate_level_agent.invoke({
        "messages": [
            HumanMessage(content=data["question"])
        ]
    })

    final_message = result["messages"][-1].content

    # Gemini may return content as a list of blocks
    if isinstance(final_message, list):
        final_message = "\n".join(
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in final_message
        )

    return final_message


clean_agent = RunnableLambda(run_agent)


# FastAPI application
app = FastAPI(
    title="Gate-Level Verilog AI Agent",
    version="1.0"
)


# LangServe route
add_routes(
    app,
    clean_agent,
    path="/verilog",
    input_type=VerilogQuestion,
    output_type=str
)
