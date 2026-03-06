"""
Main agent entry point for the LangGraph-based incident investigation system.
This replaces the traditional multi-agent orchestration with a LangGraph workflow.
"""

import dispatch_agents
from dispatch_agents import BasePayload
from investigation_graph import SimpleInvestigationWorkflow


class IncidentPayload(BasePayload):
    incident_id1: str


print("I am the stress test version")


@dispatch_agents.on(topic="dd_incident")
async def trigger(payload: IncidentPayload) -> str:
    print(
        f"[AGENT] Starting LangGraph investigation for incident {payload.incident_id1}"
    )

    # Create and run the investigation workflow
    workflow = SimpleInvestigationWorkflow()
    final_report = await workflow.investigate_incident(payload.incident_id1)

    print(f"[AGENT] Investigation completed for incident {payload.incident_id1}")

    # Return the markdown report
    return final_report.report_markdown
