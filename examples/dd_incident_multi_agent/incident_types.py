"""
Shared data structures for the incident investigation system.
"""

from enum import Enum

from pydantic import BaseModel, Field


class HypothesisStatus(Enum):
    """Status of a hypothesis investigation"""

    INCONCLUSIVE = "inconclusive"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class IncidentHypothesis(BaseModel):
    """A hypothesis about the incident root cause"""

    id: int = Field(description="Unique identifier for the hypothesis")
    hypothesis: str = Field(description="The hypothesis statement")
    investigation_report: str = Field(
        default="", description="Detailed investigation findings"
    )
    status: HypothesisStatus = Field(default=HypothesisStatus.INCONCLUSIVE)


class IncidentReport(BaseModel):
    """Final incident investigation report"""

    incident_id: str = Field(description="The incident ID")
    report_markdown: str = Field(description="Complete report in markdown format")
    confirmed_hypotheses: list[IncidentHypothesis] = Field(default_factory=list)
    rejected_hypotheses: list[IncidentHypothesis] = Field(default_factory=list)
    inconclusive_hypotheses: list[IncidentHypothesis] = Field(default_factory=list)


class InvestigationState(BaseModel):
    """State object for LangGraph investigation workflow"""

    incident_id: str
    incident_context: str | None = None
    hypotheses: list[IncidentHypothesis] = Field(default_factory=list)
    final_report: IncidentReport | None = None
    error_message: str | None = None
    thread_id: str = Field(description="Thread ID for agent continuity")
