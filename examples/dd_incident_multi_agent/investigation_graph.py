"""
Simplified LangGraph-based incident investigation workflow.
Uses two agent types: regular LLM for reasoning, dispatch agent for MCP data.
"""

import asyncio
import uuid
from typing import Any

from dispatch_agent_wrapper import DispatchAgentWrapper
from incident_types import (
    HypothesisStatus,
    IncidentHypothesis,
    IncidentReport,
    InvestigationState,
)
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class HypothesisStringList(BaseModel):
    """Structured output for hypothesis generation"""

    hypotheses: list[str] = Field(
        description="A list of 3 distinct, testable hypotheses about the incident root cause"
    )


def parse_hypotheses_from_string(text: str) -> list[str]:
    """Parse hypotheses from LLM response, handling numbered string format.

    Some models return numbered strings like:
    "1. First hypothesis\n2. Second hypothesis\n3. Third hypothesis"
    instead of proper JSON arrays.
    """
    import re

    # Try to find numbered items (1. xxx, 2. xxx, etc.)
    pattern = r"(?:^|\n)\s*(\d+)[.\)]\s*(.+?)(?=(?:\n\s*\d+[.\)]|\Z))"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return [match[1].strip() for match in matches[:3]]

    # Fallback: split by newlines and take non-empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Remove any leading numbers/bullets
    cleaned = []
    for line in lines[:3]:
        cleaned_line = re.sub(r"^[\d\-\*\.\)]+\s*", "", line)
        if cleaned_line:
            cleaned.append(cleaned_line)

    return cleaned if cleaned else [text]


class SimpleInvestigationWorkflow:
    """Simplified LangGraph workflow using two agent types"""

    def __init__(self, llm_model: str = "gpt-5.2"):
        # ChatOpenAI routes through sidecar proxy → backend
        # OPENAI_BASE_URL is set automatically by grpc_listener.py
        self.llm = ChatOpenAI(model=llm_model)
        self.datadog_agent = DispatchAgentWrapper(
            agent="dd-logs-agentic-search", function="trigger", timeout=120
        )  # For MCP tasks
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the simplified LangGraph workflow"""
        workflow = StateGraph(InvestigationState)

        # Add nodes
        workflow.add_node("gather_context", self._gather_incident_context)
        workflow.add_node("generate_hypotheses", self._generate_hypotheses)
        workflow.add_node(
            "investigate_hypotheses", self._investigate_hypotheses_parallel
        )
        workflow.add_node("generate_report", self._generate_final_report)

        # Add edges
        workflow.set_entry_point("gather_context")
        workflow.add_edge("gather_context", "generate_hypotheses")
        workflow.add_edge("generate_hypotheses", "investigate_hypotheses")
        workflow.add_edge("investigate_hypotheses", "generate_report")
        workflow.add_edge("generate_report", END)

        return workflow.compile(checkpointer=self.memory)

    async def _gather_incident_context(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Node: Gather incident context using dispatch agent (needs MCP data)"""
        print(f"[CONTEXT] Gathering context for incident {state.incident_id}")

        system_prompt = f"""You are an expert incident response analyst tasked with gathering comprehensive context about incident {state.incident_id}.

Your goal is to collect and analyze all available information about this incident to provide a complete picture for further investigation.

TASKS:
1. Get detailed incident information (status, severity, timeline, description)
2. Search for related logs around the incident timeframe
3. Look for relevant metrics that might show anomalies
4. Check for any related traces or APM data
5. Identify affected services and infrastructure
6. Note any patterns or correlations in the data

IMPORTANT GUIDELINES:
- Be thorough but focused - gather the most relevant information
- Look for time correlations around the incident start time
- Pay attention to error patterns, performance degradations, or anomalies
- Include specific data points, timestamps, and metrics values
- Organize findings logically for easy analysis

CRITICAL - DATADOG QUERY SYNTAX:
When using Datadog search tools (logs, APM, etc.), you MUST use proper query syntax. A bare "*" is NOT valid!

Valid query examples:
- "service:myservice" - Filter by service name
- "status:error" - Filter by log status
- "@http.status_code:>=500" - Filter by facet value
- "host:myhost AND status:warn" - Combine filters
- "env:production @duration:>1s" - Filter by environment and duration

Always specify at least one filter like service, host, env, status, or use the incident details to construct a meaningful query.

IMPORTANT - AVOID THESE COMMON MISTAKES:
1. DO NOT use SQL syntax (ILIKE, LIKE, SELECT, WHERE) - Datadog uses its own query language
2. DO NOT use invocation IDs or UUIDs as trace IDs - Datadog trace IDs are 32 lowercase hex characters or decimal digits
3. DO NOT use placeholder/fake trace IDs like all-zeros
4. DO NOT search for specific trace IDs unless you have a real Datadog trace ID from APM data
5. Instead of trace ID searches, use service/status/time filters to find relevant data

Provide a comprehensive summary that includes:
- Incident timeline and current status
- Affected services/systems
- Key symptoms and error patterns
- Relevant metrics and their values
- Notable log entries
- Any immediate observations about potential causes

Be concise but thorough - this context will be used to generate investigation hypotheses."""

        try:
            response = await self.datadog_agent.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    (
                        "user",
                        f"Please gather comprehensive context for incident {state.incident_id}",
                    ),
                ],
                config={"configurable": {"thread_id": f"context_{state.thread_id}"}},
            )

            incident_context = response["messages"][-1].content
            print(f"[CONTEXT] Got context: {len(incident_context)} characters")
            return {"incident_context": incident_context}

        except Exception as e:
            error_msg = f"Failed to gather incident context: {str(e)}"
            print(f"[CONTEXT] Error: {error_msg}")
            return {"error_message": error_msg}

    async def _generate_hypotheses(self, state: InvestigationState) -> dict[str, Any]:
        """Node: Generate hypotheses using regular LLM (pure reasoning, no MCP needed)"""
        print("[HYPOTHESES] Generating hypotheses")

        if state.error_message:
            return {}

        system_prompt = """You are an expert incident analyst specializing in root cause hypothesis generation.

Your task is to analyze the provided incident context and generate exactly 3 distinct, testable hypotheses about the potential root cause of the incident.

HYPOTHESIS REQUIREMENTS:
1. Each hypothesis should be specific and actionable
2. Hypotheses should cover different potential areas (e.g., infrastructure, application code, external dependencies)
3. Each hypothesis should be testable using available monitoring data, logs, metrics, and traces
4. Hypotheses should be mutually exclusive where possible
5. Focus on the most likely causes based on the symptoms observed

Format your response as a numbered list:
1. [First hypothesis]
2. [Second hypothesis]
3. [Third hypothesis]"""

        hypothesis_texts: list[str] = []

        try:
            # Try structured output first
            structured_llm = self.llm.with_structured_output(HypothesisStringList)

            result = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    (
                        "user",
                        f"Based on this incident context, generate exactly 3 investigation hypotheses:\n\n{state.incident_context}",
                    ),
                ]
            )
            hypothesis_texts = result.hypotheses[:3]
            print(
                f"[HYPOTHESES] Structured output succeeded: {len(hypothesis_texts)} hypotheses"
            )

        except Exception as structured_error:
            # Fallback: use regular LLM and parse the response
            print(
                f"[HYPOTHESES] Structured output failed ({structured_error}), trying fallback..."
            )

            try:
                response = await self.llm.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        (
                            "user",
                            f"Based on this incident context, generate exactly 3 investigation hypotheses as a numbered list:\n\n{state.incident_context}",
                        ),
                    ]
                )
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                hypothesis_texts = parse_hypotheses_from_string(response_text)
                print(
                    f"[HYPOTHESES] Fallback parsing succeeded: {len(hypothesis_texts)} hypotheses"
                )

            except Exception as fallback_error:
                error_msg = f"Failed to generate hypotheses: {fallback_error}"
                print(f"[HYPOTHESES] Error: {error_msg}")
                return {"error_message": error_msg}

        # Convert to IncidentHypothesis objects
        hypotheses = []
        for i, hypothesis_text in enumerate(hypothesis_texts[:3]):
            hypotheses.append(
                IncidentHypothesis(
                    id=i + 1,
                    hypothesis=hypothesis_text,
                    investigation_report="",
                    status=HypothesisStatus.INCONCLUSIVE,
                )
            )

        print(f"[HYPOTHESES] Generated {len(hypotheses)} hypotheses")
        return {"hypotheses": hypotheses}

    async def _investigate_single_hypothesis(
        self, hypothesis: IncidentHypothesis, incident_context: str, thread_id: str
    ) -> IncidentHypothesis:
        """Investigate a single hypothesis using dispatch agent (needs MCP data)"""
        print(
            f"[INVESTIGATE] Starting hypothesis {hypothesis.id}: {hypothesis.hypothesis[:100]}..."
        )

        hypothesis_thread_id = f"{thread_id}_hypothesis_{hypothesis.id}"

        system_prompt = f"""You are an expert incident investigator tasked with testing this specific hypothesis:

HYPOTHESIS: {hypothesis.hypothesis}

INCIDENT CONTEXT:
{incident_context}

Your goal is to determine if this hypothesis is:
- CONFIRMED: Strong evidence supports this as the root cause
- REJECTED: Evidence clearly contradicts this hypothesis
- INCONCLUSIVE: Insufficient or conflicting evidence

INVESTIGATION APPROACH:
1. Look for specific evidence that supports or contradicts the hypothesis
2. Query relevant logs, metrics, traces, and other data sources
3. Check for timing correlations with the incident
4. Look for similar patterns or precedents
5. Consider alternative explanations for any evidence found

CRITICAL - DATADOG QUERY SYNTAX:
When using Datadog search tools (logs, APM, RUM, etc.), you MUST use proper query syntax. A bare "*" is NOT valid!

Valid query examples:
- "service:myservice" - Filter by service name
- "status:error" - Filter by log status
- "@http.status_code:>=500" - Filter by facet value
- "host:myhost AND status:warn" - Combine filters
- "env:production @duration:>1s" - Filter by environment and duration
- "source:nginx status:error" - Filter by log source and status

Always construct queries using specific filters based on the incident context (service names, hosts, environments, error types, etc.). Never use just "*" as the filter.

IMPORTANT - AVOID THESE COMMON MISTAKES:
1. DO NOT use SQL syntax (ILIKE, LIKE, SELECT, WHERE) - Datadog uses its own query language
2. DO NOT use invocation IDs or UUIDs as trace IDs - Datadog trace IDs are 32 lowercase hex characters (e.g., "abc123def456...") or decimal digits
3. DO NOT use placeholder/fake trace IDs like all-zeros
4. DO NOT search for specific trace IDs unless you have a real Datadog trace ID from APM data
5. Instead of trace ID searches, use service/status/time filters to find relevant data

You MUST format your final response as:
CONCLUSION: [CONFIRMED/REJECTED/INCONCLUSIVE]

INVESTIGATION REPORT:
[Your detailed findings and reasoning with specific evidence and data points]"""

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = await self.datadog_agent.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        (
                            "user",
                            "Please investigate this hypothesis thoroughly and provide your conclusion with detailed reasoning.",
                        ),
                    ],
                    config={
                        "configurable": {
                            "thread_id": f"{hypothesis_thread_id}_attempt_{attempt}"
                        }
                    },
                )

                result = response["messages"][-1].content

                # Parse the result to extract conclusion and report
                lines = result.split("\n")
                conclusion_line = next(
                    (line for line in lines if line.startswith("CONCLUSION:")), None
                )

                if conclusion_line:
                    conclusion = (
                        conclusion_line.replace("CONCLUSION:", "").strip().upper()
                    )
                    report_start = (
                        next(
                            (
                                i
                                for i, line in enumerate(lines)
                                if "INVESTIGATION REPORT:" in line
                            ),
                            0,
                        )
                        + 1
                    )
                    investigation_report = "\n".join(lines[report_start:]).strip()

                    # Update hypothesis status
                    if conclusion == "CONFIRMED":
                        hypothesis.status = HypothesisStatus.CONFIRMED
                    elif conclusion == "REJECTED":
                        hypothesis.status = HypothesisStatus.REJECTED
                    else:
                        hypothesis.status = HypothesisStatus.INCONCLUSIVE

                    hypothesis.investigation_report = investigation_report

                    # If conclusive or last attempt, return result
                    if (
                        hypothesis.status != HypothesisStatus.INCONCLUSIVE
                        or attempt == max_retries
                    ):
                        print(
                            f"[INVESTIGATE] Completed hypothesis {hypothesis.id}: {hypothesis.status.value}"
                        )
                        return hypothesis

                else:
                    # Fallback if format not followed
                    hypothesis.investigation_report = result
                    hypothesis.status = HypothesisStatus.INCONCLUSIVE

            except Exception as e:
                error_msg = f"Investigation attempt {attempt + 1} failed: {str(e)}"
                hypothesis.investigation_report += f"\n\nERROR: {error_msg}"

                if attempt == max_retries:
                    hypothesis.status = HypothesisStatus.INCONCLUSIVE
                    print(
                        f"[INVESTIGATE] Hypothesis {hypothesis.id} failed after {max_retries + 1} attempts"
                    )

        return hypothesis

    async def _investigate_hypotheses_parallel(
        self, state: InvestigationState
    ) -> dict[str, Any]:
        """Node: Investigate all hypotheses in parallel using dispatch agent"""
        print(
            f"[INVESTIGATE_ALL] Starting parallel investigation of {len(state.hypotheses)} hypotheses"
        )

        if state.error_message or not state.hypotheses:
            return {}

        # Run investigations in parallel
        tasks = [
            self._investigate_single_hypothesis(
                hypothesis, state.incident_context or "", state.thread_id
            )
            for hypothesis in state.hypotheses
        ]

        try:
            investigated_hypotheses = await asyncio.gather(*tasks)
            print("[INVESTIGATE_ALL] Completed all investigations")
            return {"hypotheses": investigated_hypotheses}

        except Exception as e:
            error_msg = f"Failed during hypothesis investigation: {str(e)}"
            print(f"[INVESTIGATE_ALL] Error: {error_msg}")
            return {"error_message": error_msg}

    async def _generate_final_report(self, state: InvestigationState) -> dict[str, Any]:
        """Node: Generate report using regular LLM (pure reasoning, no MCP needed)"""
        print("[REPORT] Generating final report")

        if state.error_message:
            error_report = IncidentReport(
                incident_id=state.incident_id,
                report_markdown=f"# Investigation Error\n\nThe investigation could not be completed due to an error:\n\n{state.error_message}",
            )
            return {"final_report": error_report}

        # Categorize hypotheses by status
        confirmed = [
            h for h in state.hypotheses if h.status == HypothesisStatus.CONFIRMED
        ]
        rejected = [
            h for h in state.hypotheses if h.status == HypothesisStatus.REJECTED
        ]
        inconclusive = [
            h for h in state.hypotheses if h.status == HypothesisStatus.INCONCLUSIVE
        ]

        # Generate report using regular LLM
        report_prompt = f"""
        Generate a comprehensive incident investigation report based on the following information:

        INCIDENT ID: {state.incident_id}

        INCIDENT CONTEXT:
        {state.incident_context}

        INVESTIGATION RESULTS:

        CONFIRMED HYPOTHESES ({len(confirmed)}):
        {chr(10).join([f"- {h.hypothesis}: {h.investigation_report[:200]}..." for h in confirmed])}

        REJECTED HYPOTHESES ({len(rejected)}):
        {chr(10).join([f"- {h.hypothesis}: {h.investigation_report[:200]}..." for h in rejected])}

        INCONCLUSIVE HYPOTHESES ({len(inconclusive)}):
        {chr(10).join([f"- {h.hypothesis}: {h.investigation_report[:200]}..." for h in inconclusive])}

        Please create a comprehensive markdown report that includes:
        1. Executive Summary
        2. Incident Overview
        3. Investigation Methodology
        4. Key Findings
        5. Root Cause Analysis
        6. Recommendations
        7. Appendix with detailed hypothesis investigations

        Make the report professional, actionable, and suitable for both technical and non-technical audiences.
        """

        try:
            report_content = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are an expert incident analyst creating a comprehensive investigation report."
                    ),
                    ("user", report_prompt),
                ]
            )

            # Extract content from response
            report_text = (
                report_content.content
                if hasattr(report_content, "content")
                else str(report_content)
            )

            final_report = IncidentReport(
                incident_id=state.incident_id,
                report_markdown=report_text,
                confirmed_hypotheses=confirmed,
                rejected_hypotheses=rejected,
                inconclusive_hypotheses=inconclusive,
            )

            print(f"[REPORT] Generated final report: {len(report_text)} characters")
            return {"final_report": final_report}

        except Exception as e:
            error_msg = f"Failed to generate final report: {str(e)}"
            print(f"[REPORT] Error: {error_msg}")
            return {"error_message": error_msg}

    async def investigate_incident(self, incident_id: str) -> IncidentReport:
        """
        Run the complete investigation workflow for an incident.

        Args:
            incident_id: The incident ID to investigate

        Returns:
            IncidentReport: Comprehensive investigation report
        """
        # Create initial state
        thread_id = f"investigation_{incident_id}_{uuid.uuid4().hex[:8]}"
        initial_state = InvestigationState(incident_id=incident_id, thread_id=thread_id)

        # Run the workflow
        config = {
            "configurable": {"thread_id": f"investigation_workflow_{incident_id}"}
        }

        final_state = await self.graph.ainvoke(initial_state, config=config)

        if final_state.get("final_report"):
            return final_state["final_report"]
        else:
            # Fallback error report
            return IncidentReport(
                incident_id=incident_id,
                report_markdown=f"# Investigation Failed\n\nThe investigation for incident {incident_id} could not be completed.\n\nError: {final_state.get('error_message', 'Unknown error')}",
            )
