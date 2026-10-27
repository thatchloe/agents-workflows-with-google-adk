"""Helpers for executing composed ADK agents through the ADK Runner."""

import json
import re
from typing import Any, Dict, List, Tuple
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

APP_NAME = "research_assistant"
USER_ID = "researcher"


async def run_adk_agent(
    agent: BaseAgent,
    message: str,
    initial_state: Dict[str, Any] = None
) -> Tuple[List[Event], Dict[str, Any]]:
    """Run an ADK agent (LoopAgent, SequentialAgent, ...) via agent.run_async().

    The Runner owns execution: workflow agents decide ordering, concurrency
    and termination. Results are read back from the events and session state.

    Args:
        agent: Root agent to execute
        message: User message that starts the invocation
        initial_state: Session state available to agents and callbacks

    Returns:
        Tuple of (all events emitted, final session state)
    """
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state=initial_state or {}
    )

    events = []
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)])
    ):
        events.append(event)

    session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session.id
    )
    return events, dict(session.state)


def parse_json_output(text: Any) -> Dict[str, Any]:
    """Parse an agent's JSON output, tolerating markdown code fences."""
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip())
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}


def event_text(event: Event) -> str:
    """Concatenate the non-thought text parts of an event."""
    if not event.content or not event.content.parts:
        return ""
    return "".join(p.text for p in event.content.parts if p.text and not p.thought)
