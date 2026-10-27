import json
from typing import Dict, Any, List
from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google import genai
from google.genai.types import GenerateContentConfig
from agents.adk_execution import run_adk_agent, parse_json_output, event_text


class ResearcherAgent(LlmAgent):
    """
    Generator agent: Answers research questions or refines based on feedback.

    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize the researcher agent.

        Args:
            model: Gemini model name
        """
        instruction = """You are a research assistant that answers questions accurately and thoroughly.

Your role:
1. Answer the research question clearly and comprehensively
2. If you receive feedback from the critic, improve your previous answer addressing all concerns
3. Cite reasoning and provide evidence where possible
4. Include 3-5 key points with supporting details
5. Mention relevant sources or areas of research

Output format (JSON):
{
  "answer": "Your comprehensive answer here with evidence and reasoning",
  "key_points": ["point1 with evidence", "point2 with evidence", "point3 with evidence"],
  "sources_mentioned": ["source1", "source2", "source3"],
  "confidence": "high/medium/low",
  "iteration_notes": "What you improved this iteration (if applicable)"
}

Focus on accuracy, clarity, and continuous improvement. Each iteration should show measurable progress."""

        # Initialize ADK LlmAgent
        super().__init__(
            name="researcher",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
                response_mime_type="application/json"
            )
        )

    def generate(self, client: genai.Client, query: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate research answer using direct genai.Client call.

        This is the execution part - use agent's config but execute directly.

        Args:
            client: Configured genai.Client
            query: Research question
            context: Previous conversation history (for refinement)

        Returns:
            Dictionary with answer and metadata
        """
        # Build prompt with context (mimics ADK context passing)
        if context:
            prompt = f"{self.instruction}\n\nConversation history:\n"
            for msg in context:
                prompt += f"\n{msg['role']}: {msg['content']}"
            prompt += f"\n\nuser: {query}"
        else:
            prompt = f"{self.instruction}\n\nuser: {query}"

        # Direct execution using genai.Client (part)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.generate_content_config
        )

        # Parse JSON response
        try:
            result = json.loads(response.text)
            result['_metadata'] = {
                'agent': self.name,
                'model': self.model,
                'execution': 'direct_genai_client'
            }
            return result
        except json.JSONDecodeError:
            return {
                'answer': response.text,
                'key_points': [],
                'sources_mentioned': [],
                'confidence': 'low',
                'iteration_notes': 'JSON parsing failed',
                '_metadata': {
                    'agent': self.name,
                    'error': 'json_parse_error'
                }
            }


class ResearchCriticAgent(LlmAgent):
    """
    Validator agent: Evaluates answer quality and provides feedback.

    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """Initialize the critic agent.

        Args:
            model: Gemini model name
        """
        instruction = """You are a research quality critic that evaluates answers.

Your role:
1. Evaluate the answer for accuracy, completeness, and clarity
2. Assign a quality level: excellent, good, needs_improvement, or poor
3. Provide specific, actionable feedback for improvement
4. Decide if answer is good enough to stop, or needs another iteration
5. Calculate a quality score (0-1) based on your assessment

Quality criteria:
- Excellent (0.90-1.00): Thorough, accurate, well-structured, comprehensive evidence
- Good (0.80-0.89): Accurate and complete, all key points covered
- Needs Improvement (0.50-0.79): Missing key points, needs more evidence or clarity
- Poor (0.00-0.49): Incomplete, unclear, or potentially inaccurate

Output format (JSON):
{
  "quality": "excellent/good/needs_improvement/poor",
  "quality_score": 0.85,
  "feedback": "Specific feedback for improvement",
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "should_stop": true/false,
  "reasoning": "Why to stop or continue"
}

Set should_stop=true ONLY if quality_score >= 0.80.
Otherwise set should_stop=false to trigger another iteration."""

        # Initialize ADK LlmAgent
        super().__init__(
            name="critic",
            model=model,
            instruction=instruction,
            generate_content_config=GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,  # headroom for thinking models used as critic
                response_mime_type="application/json"
            )
        )

    def evaluate(self, client: genai.Client, question: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate research answer quality using direct genai.Client call.

        This is the execution part - use agent's config but execute directly.

        Args:
            client: Configured genai.Client
            question: Original research question
            answer: Answer dictionary from researcher

        Returns:
            Evaluation dictionary with should_stop flag
        """
        prompt = f"""{self.instruction}

user: Please evaluate this research answer:

Question: {question}

Answer: {answer.get('answer', 'No answer provided')}
Key Points: {json.dumps(answer.get('key_points', []))}
Confidence: {answer.get('confidence', 'unknown')}"""

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.generate_content_config
        )

        try:
            result = json.loads(response.text)
            result['_metadata'] = {
                'agent': self.name,
                'model': self.model,
                'execution': 'direct_genai_client'
            }
            return result
        except json.JSONDecodeError:
            return {
                'quality': 'needs_improvement',
                'quality_score': 0.5,
                'feedback': 'Unable to parse evaluation',
                'strengths': [],
                'weaknesses': ['Evaluation failed'],
                'should_stop': False,
                'reasoning': 'JSON parsing failed',
                '_metadata': {
                    'agent': self.name,
                    'error': 'json_parse_error'
                }
            }


# ----------------------------------------------------------------------------
# LoopAgent callbacks: logging plus the critic's escalate (termination) decision
# ----------------------------------------------------------------------------

def _researcher_started(callback_context: CallbackContext) -> None:
    iteration = callback_context.state.get('loop_iteration', 0) + 1
    callback_context.state['loop_iteration'] = iteration
    max_iterations = callback_context.state.get('max_iterations', '?')
    print(f"\n   Iteration {iteration}/{max_iterations}")
    print(f"      → researcher generating answer...")


def _researcher_finished(callback_context: CallbackContext) -> None:
    answer = parse_json_output(callback_context.state.get('research_answer'))
    print(f"      ✓ Answer generated (confidence: {answer.get('confidence', 'unknown')})")


def _critic_started(callback_context: CallbackContext) -> None:
    print(f"      → critic evaluating quality...")


def _critic_decides(callback_context: CallbackContext) -> None:
    """Escalate out of the LoopAgent once the critic's score meets the threshold."""
    evaluation = parse_json_output(callback_context.state.get('critic_evaluation'))
    quality_score = float(evaluation.get('quality_score', 0.5))
    threshold = callback_context.state.get('quality_threshold', 0.80)

    print(f"      ✓ Quality: {evaluation.get('quality', 'unknown')} (score: {quality_score:.2f})")

    # Writing state guarantees ADK emits an event carrying these actions
    callback_context.state['last_quality_score'] = quality_score
    if quality_score >= threshold:
        print(f"      Quality threshold met - critic escalates, LoopAgent terminates")
        callback_context.actions.escalate = True
    else:
        print(f"      Quality below threshold - Continue refining...")
        print(f"      Feedback: {str(evaluation.get('feedback', 'No feedback'))[:80]}...")


def create_research_loop_agent(model: str = "gemini-2.0-flash",
                                max_iterations: int = 3,
                                critic_model: str = None) -> LoopAgent:
    """
    Creates a LoopAgent for iterative research refinement.

    Args:
        model: Gemini model to use
        max_iterations: Maximum loop iterations (safety limit)
        critic_model: Optional stricter model for the critic (defaults to model)

    Returns:
        LoopAgent configured for research refinement
    """
    # TODO 1: Create the two agents for the LoopAgent
    #
    # Create the generator and validator agents:
    # - ResearcherAgent: The generator that creates/improves answers
    # - ResearchCriticAgent: The validator that evaluates quality

    researcher = ResearcherAgent(model=model)
    critic = ResearchCriticAgent(model=critic_model or model)

    # Each agent's final output is stored in session state; the critic's
    # after_agent_callback escalates to end the loop when quality is sufficient
    researcher.output_key = "research_answer"
    researcher.before_agent_callback = _researcher_started
    researcher.after_agent_callback = _researcher_finished
    critic.output_key = "critic_evaluation"
    critic.before_agent_callback = _critic_started
    critic.after_agent_callback = _critic_decides

    # TODO 2: Compose agents into LoopAgent
    #
    # Create a LoopAgent that will run the researcher and critic iteratively.
    #

    refinement_loop = LoopAgent(
        name="research_refinement_loop",
        sub_agents=[researcher, critic],
        max_iterations=max_iterations
    )

    return refinement_loop


async def execute_research_loop(
    client: genai.Client,
    query: str,
    max_iterations: int = 3,
    model: str = "gemini-2.0-flash",
    sources: Dict[str, Any] = None,
    quality_threshold: float = 0.80,
    critic_model: str = None
) -> Dict[str, Any]:
    """
    Execute iterative research refinement using ADK LoopAgent.

    Args:
        client: Configured genai.Client
        query: Research question
        max_iterations: Maximum loop iterations
        model: Gemini model name
        sources: Output of execute_source_gathering, used as grounding context
        quality_threshold: Critic score required to stop refining
        critic_model: Optional stricter model for the critic (defaults to model)

    Returns:
        Dictionary with final answer and iteration history
    """
    print(f"\n Research Loop: {query[:60]}...")
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Quality Threshold: {quality_threshold:.2f}")

    # Ground the researcher in the sources gathered in Stage 2
    research_prompt = query
    top_sources = (sources or {}).get('aggregated_sources', {}).get('top_sources', [])
    if top_sources:
        source_lines = "\n".join(
            f"- [{s.get('type', 'source')}] {s.get('title', 'Untitled')}: {s.get('snippet', '')}"
            for s in top_sources
        )
        research_prompt = f"{query}\n\nUse these gathered sources as evidence:\n{source_lines}"
        print(f"   Grounded with {len(top_sources)} sources from source gathering")

    # Create LoopAgent
    loop_agent = create_research_loop_agent(
        model=model,
        max_iterations=max_iterations,
        critic_model=critic_model
    )

    print(f"   Created LoopAgent: {loop_agent.name}")
    print(f"   Type: {type(loop_agent).__name__}")
    print(f"   Sub-agents: {len(loop_agent.sub_agents)} (researcher + critic)")

    print(f"\n   🔄 Executing LoopAgent via ADK Runner (loop_agent.run_async())...")

    # The LoopAgent controls iteration: it runs researcher -> critic until the
    # critic escalates or max_iterations is reached. Earlier drafts and critic
    # feedback reach the researcher through the shared session history.
    events, _ = await run_adk_agent(
        loop_agent,
        message=research_prompt,
        initial_state={
            'quality_threshold': quality_threshold,
            'max_iterations': max_iterations
        }
    )

    # Rebuild the iteration history from the events the agents emitted
    iteration_history = []
    answer = None
    escalated = False
    for event in events:
        if event.actions.escalate:
            escalated = True
            if iteration_history:
                iteration_history[-1]['should_stop'] = True
        text = event_text(event)
        if not text or event.partial:
            continue
        if event.author == 'researcher':
            answer = parse_json_output(text) or {'answer': text, 'confidence': 'low'}
        elif event.author == 'critic' and answer is not None:
            iteration_history.append({
                'iteration': len(iteration_history) + 1,
                'answer': answer,
                'evaluation': parse_json_output(text),
                'should_stop': False
            })
            answer = None

    if not escalated:
        print(f"\n   Max iterations reached - Returning best attempt")
    final_answer = iteration_history[-1]['answer'] if iteration_history else {}

    print(f"   LoopAgent execution completed ({len(iteration_history)} iterations)")

    return {
        'query': query,
        'final_answer': final_answer,
        'iterations_run': len(iteration_history),
        'iteration_history': iteration_history,
        'loop_agent': loop_agent,  # Include the actual LoopAgent object
        'pattern': 'ADK LoopAgent',
        'execution_mode': 'adk_runner'
    }
