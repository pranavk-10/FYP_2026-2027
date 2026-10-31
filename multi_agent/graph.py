import os
import sys

MULTI_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if MULTI_AGENT_DIR not in sys.path:
    sys.path.insert(0, MULTI_AGENT_DIR)

from langgraph.graph import StateGraph, START, END
from state import DiagnosticState
from nodes.advocate import advocate_node
from nodes.skeptic import skeptic_node
from nodes.evidence import evidence_checker_node
from nodes.moderator import moderator_node


def route_debate(state: DiagnosticState):
    verdict = state.get("verdict")
    # Hard stop: the moderator increments current_round after each round, so once it
    # exceeds max_rounds the debate ends even if the LLM asked to continue.
    rounds_done = state.get("current_round", 1) > state.get("max_rounds", 2)
    if verdict is None or verdict.action == "finalize" or rounds_done:
        return END
    return "advocate"

workflow = StateGraph(DiagnosticState)
workflow.add_node("advocate", advocate_node)
workflow.add_node("skeptic", skeptic_node)
workflow.add_node("evidence", evidence_checker_node)
workflow.add_node("moderator", moderator_node)

workflow.add_edge(START, "advocate")
workflow.add_edge("advocate", "skeptic")
workflow.add_edge("skeptic", "evidence")
workflow.add_edge("evidence", "moderator")
workflow.add_conditional_edges("moderator", route_debate, {END: END, "advocate": "advocate"})

app = workflow.compile()