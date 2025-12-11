"""
LangGraph agent: state, nodes, and graph assembly.

This module contains:
- AgentState: TypedDict for agent state
- HistoryEntry: Dataclass for iteration history
- Agent nodes: sql_generator, evaluator, duckdb_node, controller
- Graph assembly: build_sql_agent_graph()
- Runner: run_sql_agent() convenience function
"""

import os
import uuid
from dataclasses import dataclass
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
import logging

logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent_sql.core import ScenarioConfig, DiffResult, run_and_compare
from agent_sql.prompts import (
    SQLGeneratorPromptBuilder,
    EvaluatorPromptBuilder,
    format_diff_for_prompt,
)


# ============================================================================
# Agent State
# ============================================================================

@dataclass
class HistoryEntry:
    """A single entry in the iteration history."""
    iteration: int
    sql: str
    diff_summary: str
    success: bool


class AgentState(TypedDict):
    """State for the SQL generation agent."""
    # Scenario configuration (constant through the run)
    scenario: ScenarioConfig
    
    # Current iteration
    iteration: int
    
    # Current SQL attempt
    sql: Optional[str]
    
    # Last diff result
    last_diff: Optional[DiffResult]
    
    # Expert evaluation of the failure (from evaluator node)
    evaluation: Optional[str]
    
    # History of attempts (for logging, not sent to LLM)
    history: list[HistoryEntry]
    
    # Terminal flags
    success: bool
    done: bool
    reason: Optional[str]


# ============================================================================
# Agent Nodes
# ============================================================================

# Initialize prompt builders (can be swapped for different implementations)
_sql_prompt_builder = SQLGeneratorPromptBuilder()
_evaluator_prompt_builder = EvaluatorPromptBuilder()

# Initialize the LLM
# Allow user to override model via environment variable (e.g. OPENAI_MODEL=gpt-5.1)
_model_name = os.getenv("OPENAI_MODEL", "gpt-4o")

# o1 models do not support temperature
if _model_name.startswith("o1"):
    _llm = ChatOpenAI(model=_model_name)
else:
    _llm = ChatOpenAI(model=_model_name, temperature=0.5)

_evaluator_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5)


def sql_generator(state: AgentState) -> dict:
    """
    Generate or refine SQL based on the scenario and previous feedback.
    
    This node:
    1. Builds the prompt with scenario context and previous diff (if any)
    2. Calls the LLM to generate SQL
    3. Cleans the response (strips markdown, etc.)
    4. Returns updated state with new SQL
    """
    scenario = state["scenario"]
    iteration = state["iteration"]
    previous_sql = state.get("sql")
    last_diff = state.get("last_diff")
    evaluation = state.get("evaluation")
    
    # Build the prompt
    system_prompt, user_prompt = _sql_prompt_builder.build_prompt(
        scenario=scenario,
        previous_sql=previous_sql,
        last_diff=last_diff,
        evaluation=evaluation,
        iteration=iteration
    )
    
    # Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Call the LLM
    logger.info(f"\n{'='*60}")
    logger.info(f"🤖 SQL Generator - Iteration {iteration}")
    logger.info(f"{'='*60}")
    
    response = _llm.invoke(messages)
    raw_sql = response.content
    
    # Clean the SQL (remove markdown code blocks if present)
    sql = raw_sql.strip()
    if sql.startswith("```"):
        # Remove markdown code fences
        lines = sql.split("\n")
        # Remove first line (```sql or ```)
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines)
    
    logger.info(f"Generated SQL:\n{sql[:500]}{'...' if len(sql) > 500 else ''}")
    
    return {
        "sql": sql,
        "iteration": iteration + 1
    }


def evaluator_node(state: AgentState) -> dict:
    """
    An LLM evaluator that analyzes WHY the SQL failed and provides 
    specific guidance on how to fix it.
    
    This node only runs if there was a failure (not success).
    """
    scenario = state["scenario"]
    sql = state["sql"]
    last_diff = state.get("last_diff")
    
    # If we succeeded, no evaluation needed
    if not last_diff or last_diff.ok:
        return {"evaluation": None}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Evaluator - Analyzing failure")
    logger.info(f"{'='*60}")
    
    # Build evaluator prompt
    system_prompt, user_prompt = _evaluator_prompt_builder.build_prompt(
        scenario=scenario,
        sql=sql,
        last_diff=last_diff
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    evaluation = _evaluator_llm.invoke(messages)
    evaluation_text = evaluation.content
    
    logger.info(f"Evaluation:\n{evaluation_text[:500]}{'...' if len(evaluation_text) > 500 else ''}")
    
    return {"evaluation": evaluation_text}


def duckdb_node(state: AgentState) -> dict:
    """
    Execute the current SQL and compare against expected output.
    
    This node:
    1. Runs the SQL via run_and_compare()
    2. Updates last_diff with the result
    3. Sets success=True if diff.ok
    4. Appends to history for logging
    """
    scenario = state["scenario"]
    sql = state["sql"]
    iteration = state["iteration"]
    history = state.get("history", [])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔧 DuckDB Executor - Testing SQL")
    logger.info(f"{'='*60}")
    
    # Execute and compare
    diff = run_and_compare(scenario, sql)
    
    # Format for logging
    diff_summary = format_diff_for_prompt(diff)
    logger.info(diff_summary)
    
    # Create history entry
    entry = HistoryEntry(
        iteration=iteration,
        sql=sql,
        diff_summary=diff_summary,
        success=diff.ok
    )
    
    # Return updated state
    return {
        "last_diff": diff,
        "success": diff.ok,
        "history": history + [entry]
    }


def controller(state: AgentState) -> dict:
    """
    Decide whether to continue iterating or stop.
    
    Routing logic:
    - If success=True → done, route to END
    - If iteration >= max_iters → done (max iterations), route to END
    - Otherwise → route to evaluator → sql_generator
    """
    scenario = state["scenario"]
    iteration = state["iteration"]
    success = state.get("success", False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎮 Controller - Deciding next step")
    logger.info(f"{'='*60}")
    
    if success:
        logger.info("✓ SUCCESS! SQL matches expected output.")
        return {
            "done": True,
            "reason": "success"
        }
    
    if iteration >= scenario.max_iters:
        logger.info(f"✗ Max iterations ({scenario.max_iters}) reached without success.")
        return {
            "done": True,
            "reason": "max_iterations"
        }
    
    logger.info(f"→ Continuing to iteration {iteration} (max: {scenario.max_iters})")
    return {
        "done": False
    }


def route_after_controller(state: AgentState) -> Literal["evaluator", "__end__"]:
    """
    Router function for conditional edges after controller.
    """
    if state.get("done", False):
        return END
    else:
        return "evaluator"


# ============================================================================
# Graph Assembly
# ============================================================================

def build_sql_agent_graph():
    """
    Assemble the StateGraph for the SQL generation agent.
    
    Graph structure:
        START → sql_generator → duckdb_node → controller → (END or evaluator → sql_generator)
    """
    # Create the graph builder with our state type
    graph_builder = StateGraph(AgentState)
    
    # Add nodes
    graph_builder.add_node("sql_generator", sql_generator)
    graph_builder.add_node("duckdb_node", duckdb_node)
    graph_builder.add_node("evaluator", evaluator_node)
    graph_builder.add_node("controller", controller)
    
    # Add edges
    graph_builder.add_edge(START, "sql_generator")
    graph_builder.add_edge("sql_generator", "duckdb_node")
    graph_builder.add_edge("duckdb_node", "controller")
    graph_builder.add_edge("evaluator", "sql_generator")
    
    # Conditional edge from controller
    graph_builder.add_conditional_edges(
        "controller",
        route_after_controller,
        {
            "evaluator": "evaluator",
            END: END
        }
    )
    
    # Compile with memory checkpointer
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    return graph


# ============================================================================
# Runner
# ============================================================================

def run_sql_agent(scenario: ScenarioConfig, verbose: bool = True) -> dict:
    """
    Run the SQL agent on a scenario and return the final state.
    
    Args:
        scenario: The scenario configuration
        verbose: If True, print iteration details
    
    Returns:
        The final state dict with sql, success, history, etc.
    """
    # Create initial state
    initial_state: AgentState = {
        "scenario": scenario,
        "iteration": 0,
        "sql": None,
        "last_diff": None,
        "evaluation": None,
        "history": [],
        "success": False,
        "done": False,
        "reason": None
    }
    
    # Config for this run (unique thread_id)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    if verbose:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Running SQL Agent on: {scenario.name}")
        logger.info(f"# Max iterations: {scenario.max_iters}")
        logger.info(f"{'#'*70}")
    
    # Build and invoke the graph
    graph = build_sql_agent_graph()
    final_state = graph.invoke(initial_state, config={**config, "recursion_limit": 100})
    
    if verbose:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# FINAL RESULT")
        logger.info(f"{'#'*70}")
        logger.info(f"Success: {final_state.get('success', False)}")
        logger.info(f"Iterations: {final_state.get('iteration', 0)}")
        logger.info(f"Reason: {final_state.get('reason', 'unknown')}")
        
        if final_state.get("success"):
            logger.info(f"\n✓ Final SQL:\n{final_state.get('sql', 'N/A')}")
    
    return final_state

