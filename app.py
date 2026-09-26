import os
import re

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

Output formatting rules:
- Use clean plain text formatting.
- Do not use Markdown headings with # symbols.
- Do not use ** for bold text.
- Do not use triple backticks around Verilog code.
- Do not put the answer inside quotation marks.
- Present Verilog code as plain text with normal indentation.
- Use simple numbered lists when explaining steps.
- Keep the answer clear and readable.
- Separate explanations and Verilog code with blank lines.

Answer the user's actual question directly and clearly.
"""
)


# Input format
class VerilogQuestion(BaseModel):
    question: str


# Gate-Level scope checker
def is_gate_level_question(question):
    q = question.lower()

    # Reject Dataflow modeling
    dataflow_terms = [
        "dataflow",
        "data flow",
        "assign statement",
        "using assign",
        "dataflow modeling"
    ]

    # Reject Behavioral modeling
    behavioral_terms = [
        "behavioral",
        "behavioural",
        "using always",
        "always block",
        "always @",
        "always_ff",
        "always_comb"
    ]

    # Reject unrelated programming languages/topics
    unrelated_terms = [
        "java",
        "python",
        "c++",
        "javascript",
        "html",
        "css"
    ]

    if any(term in q for term in dataflow_terms):
        return False

    if any(term in q for term in behavioral_terms):
        return False

    if any(term in q for term in unrelated_terms):
        return False

    # Gate-level related terms
    gate_level_terms = [
        "gate-level",
        "gate level",
        "gate primitive",
        "and gate",
        "or gate",
        "not gate",
        "nand gate",
        "nor gate",
        "xor gate",
        "xnor gate",
        "structural verilog",
        "structural modeling",
        "structural model"
    ]

    # If explicitly Gate-Level, allow it
    if any(term in q for term in gate_level_terms):
        return True

    # Common circuit-design requests can be Gate-Level requests
    circuit_terms = [
        "adder",
        "subtractor",
        "multiplexer",
        "mux",
        "demultiplexer",
        "demux",
        "encoder",
        "decoder",
        "comparator",
        "flip-flop",
        "flip flop",
        "latch",
        "register",
        "counter",
        "parity",
        "truth table",
        "circuit diagram"
    ]

    if any(term in q for term in circuit_terms):
        return True

    return False


# Run the agent and return only the final answer
def run_agent(data):

    question = data["question"]

    # Check whether the request belongs to Gate-Level modeling
    if not is_gate_level_question(question):
        return (
            "This agent supports Gate-Level Verilog modeling only. "
            "Dataflow, Behavioral and unrelated topics are outside "
            "the scope of this agent."
        )

    result = gate_level_agent.invoke({
        "messages": [
            HumanMessage(content=question)
        ]
    })

    final_message = result["messages"][-1].content

    # Gemini may return content as a list of text blocks
    if isinstance(final_message, list):
        final_message = "\n".join(
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in final_message
        )

    # Remove Markdown formatting
    final_message = re.sub(
        r"```(?:verilog|systemverilog|v)?",
        "",
        final_message,
        flags=re.IGNORECASE
    )

    final_message = final_message.replace("```", "")

    # Remove Markdown heading symbols
    final_message = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        final_message,
        flags=re.MULTILINE
    )

    # Remove Markdown bold/italic markers
    final_message = final_message.replace("**", "")
    final_message = final_message.replace("__", "")

    # Remove inline code backticks
    final_message = re.sub(
        r"`([^`]*)`",
        r"\1",
        final_message
    )

    return final_message.strip()


# Clean agent runnable
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
