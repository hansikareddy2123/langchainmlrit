
import os
from fastapi import FastAPI
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Create the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0
)

# Create the Gate-Level Verilog Agent
gate_level_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="""You are a specialized Gate-Level Verilog AI Agent.

Answer questions specifically related to gate-level modeling in Verilog HDL.

You can:
- Explain gate-level modeling concepts.
- Explain AND, OR, NOT, NAND, NOR, XOR and XNOR gates.
- Generate gate-level Verilog code.
- Explain gate-level Verilog code.
- Debug gate-level Verilog code.
- Design digital circuits using gate primitives.
- Generate circuits such as adders, subtractors, multiplexers,
  demultiplexers, encoders, decoders and comparators.

When generating gate-level Verilog:
- Use structural gate-level modeling.
- Use Verilog gate primitives such as and, or, not, nand, nor,
  xor and xnor.
- Do not use assign statements when gate-level modeling is requested.
- Do not use always blocks when gate-level modeling is requested.
- Provide complete Verilog code when requested.
- Briefly explain the important gates and connections.
"""
)

# Create FastAPI application
app = FastAPI(
    title="Gate-Level Verilog AI Agent",
    version="1.0"
)

# Expose the agent through LangServe
add_routes(
    app,
    gate_level_agent,
    path="/verilog"
)
