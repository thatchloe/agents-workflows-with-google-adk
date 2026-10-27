# Agent Workflows with Google ADK

An AI research assistant built from cooperating agents using **Google's Agent Development Kit (ADK)** and **Gemini on Vertex AI**. You give it a research question. It classifies the question, gathers sources in parallel, drafts and refines an answer until a critic is satisfied, fact-checks and synthesizes the result, and writes a Markdown research report with citations and performance metrics.

## Project Overview

### Project DeepDive

In this project, you are the Lead AI Architect at a data intelligence firm. Your clients are senior executives who need deep, accurate technical briefings, and they need them fast.

Your VP of Engineering has just put a new ticket on your desk: **Project DeepDive**.

**The goal:** build an agentic AI research assistant. When asked a hard question, it shouldn't just guess. It should:

1. Plan where to look.
2. Read papers and websites (simulated tools in this project).
3. Draft an answer.
4. Critique its own work, and fix it if it doesn't meet the criteria.
5. Fact-check the claims.
6. Format the bibliography.

You'll use Google ADK (Agent Development Kit) to build a team of research agents that produces a research report.

### Your mission

You're given a project-starter codebase. The infrastructure (the boilerplate code) is in place, but the agent implementation is missing.

The codebase contains `TODO` markers. Your job is to add ADK agents and agentic workflow patterns so the research assistant works.

### Architecture

The assistant is a 7-stage `SequentialAgent` pipeline:

| # | Stage | ADK pattern | File |
|---|---|---|---|
| 1 | Domain classification (router) | `LlmAgent`: low temperature, JSON output | `agents/router.py`, `agents/other_agents.py` |
| 2 | Source gathering | `ParallelAgent` (web, arXiv, Scholar) inside a `SequentialAgent` with an aggregator | `agents/source_gatherer.py` |
| 3 | Research refinement | `LoopAgent`: researcher (generator) and critic (validator) | `agents/researcher.py` |
| 4 | Fact checking | `LlmAgent` | `agents/other_agents.py` |
| 5 | Synthesis | `LlmAgent` | `agents/other_agents.py` |
| 6 | Citation formatting (APA) | `LlmAgent` | `agents/other_agents.py` |
| 7 | Performance evaluation | `PerformanceEvaluator` | `agents/evaluator.py`, `agents/orchestrator.py` |

```
query ─► Router ─► ┌ web_search ─────┐
                   ├ arxiv_search ───┼─► Aggregator ─► Researcher ⇄ Critic ─► Fact Check ─► Synthesis ─► Citations ─► Metrics ─► research_report.md
                   └ scholar_search ─┘                 (loop until score ≥ threshold)
```

### Patterns

- **Iterative refinement (`LoopAgent`):** `research_refinement_loop` runs `[researcher, critic]` and is executed by the ADK `Runner` (`run_async()`), so the `LoopAgent` itself controls iteration. The critic scores each draft from 0 to 1 and gives feedback. Its `after_agent_callback` sets `escalate` once the score reaches `QUALITY_THRESHOLD`, and ADK then ends the loop; otherwise the loop runs until `MAX_ITERATIONS`. Earlier drafts and critic feedback reach the researcher through the shared ADK session history.
- **Fan-out / fan-in (`ParallelAgent` + `SequentialAgent`):** `source_gathering_workflow` is also executed by the ADK `Runner`. Its `ParallelAgent`, `parallel_source_searches`, runs the three search agents as concurrent async model calls. The `SequentialAgent` starts `source_aggregator` only after all three finish; it de-duplicates and ranks the results, which it reads from the session.
- **Deterministic routing (`LlmAgent`):** `domain_classifier` uses `temperature=0.3` and JSON output, so the same query classifies the same way on every run.
- **Grounding:** the top aggregated sources from Stage 2 go into the research loop, so the researcher writes from the gathered evidence.
- **Monitoring:** `PerformanceEvaluator` records quality, run time, source count, iterations, fact checks and citations. It reports a health status and any bottlenecks.

> **Note:** the search agents *simulate* results with the LLM. Titles and URLs are plausible but don't come from a live search engine. Treat the bibliography as illustrative.

### Project structure

```
├── main.py                # Entry point: config, Gemini client, runs the workflow, writes the report
├── agents/
│   ├── router.py          # DomainClassifierAgent / QueryComplexityAgent (LlmAgent)
│   ├── source_gatherer.py # Search agents, aggregator, ParallelAgent + SequentialAgent
│   ├── researcher.py      # ResearcherAgent, ResearchCriticAgent, LoopAgent
│   ├── adk_execution.py   # Runs composed ADK agents through InMemoryRunner
│   ├── orchestrator.py    # 7-stage workflow + report generation
│   ├── evaluator.py       # PerformanceMetrics / PerformanceEvaluator
│   └── other_agents.py    # Classifier, fact checker, synthesizer, citation agents
├── utils/config.py        # Loads settings from .env
├── .env.example           # Configuration template
├── research_report.md     # Example output (see "Example Run")
└── screenshots/           # Terminal output from the example run
```

## Setup

### Prerequisites

- Python 3.10+
- A Google Cloud project with the **Vertex AI / Agent Platform API** (`aiplatform.googleapis.com`) enabled
- A service account with the **Agent Platform User** role (`roles/aiplatform.user`). Note that *Viewer* can't call models.

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add credentials

Download a JSON key for your service account and save it in the project root as `service-account-key.json`. It is git-ignored, so never commit it.

### 3. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to the service account key | `service-account-key.json` |
| `PROJECT_ID` | Google Cloud project ID | `my-project-123` |
| `LOCATION` | Vertex AI location. Current Gemini models need `global` | `global` |
| `MODEL_NAME` | Model for all agents | `gemini-3.5-flash-lite` |
| `CRITIC_MODEL` | Optional stricter model for the research critic (defaults to `MODEL_NAME`) | `gemini-3.5-flash` |
| `MAX_ITERATIONS` | Maximum refinement passes | `3` |
| `QUALITY_THRESHOLD` | Critic score needed to stop refining | `0.95` |
| `RESEARCH_QUERY` | Optional custom query | see below |

Alternatively, set `GOOGLE_API_KEY` to use a Gemini API key instead of Vertex AI.

## Usage

```bash
source .venv/bin/activate

# Run the default sample query
python main.py

# Run your own query
export RESEARCH_QUERY="How has CRISPR gene editing changed crop breeding in agriculture?"
python main.py
```

The console shows each stage as it runs: agent creation (`Type: SequentialAgent`, `Type: ParallelAgent`, `Type: LoopAgent`), the router's classification, the parallel searches, each researcher/critic iteration with its score, and a final summary with performance metrics. The report is saved to `research_report.md`.

### Troubleshooting

| Error | Fix |
|---|---|
| `ImportError: cannot import name 'genai'` | Activate the virtual environment (`source .venv/bin/activate`) |
| `403 ... API has not been used in project` | Enable `aiplatform.googleapis.com` for the project |
| `403 Permission 'aiplatform.endpoints.predict' denied` | Grant the service account **Agent Platform User** |
| `404 Publisher model ... not found` | The model has been retired or isn't available in that location. Use a current model and `LOCATION=global` |
| 0 sources, or scores stuck at 0.50 | The model's JSON output is being cut off. Use a non-thinking model for `MODEL_NAME`, or raise the agents' `max_output_tokens` |

## Example Run

### Unique Research Query

```bash
export RESEARCH_QUERY="How has CRISPR gene editing changed crop breeding in agriculture?"
python main.py
```

Configuration: `MODEL_NAME=gemini-3.5-flash-lite`, `CRITIC_MODEL=gemini-3.5-flash`, `QUALITY_THRESHOLD=0.95`, `MAX_ITERATIONS=3`, `LOCATION=global`.

### Results

| Metric | Value |
|---|---|
| Domain (router) | biology (confidence 0.95) |
| Sources found (web / arXiv / Scholar) | 17 (6 / 5 / 6) |
| Research iterations | 2 (critic score 0.78 → 0.96) |
| Credibility score | 0.92 |
| Citations | 10 |
| Execution time | ~37 s |
| Health status | excellent |

The full report is in [`research_report.md`](research_report.md). The terminal output is in [`screenshots/`](screenshots/):

- `01_initialization.png`: configuration, client setup, and creation of the `SequentialAgent` and `ParallelAgent`
- `02_thought_process.png`: router classification, the parallel searches (fan-out and fan-in), and the `LoopAgent` researcher/critic iterations
- `03_workflow_completed.png`: the WORKFLOW COMPLETED summary and Stage 7 performance metrics
- `full_terminal_output.txt`: the complete log. The PNGs were rendered from this captured output.

### Observations

- **Router:** Classified the query as `biology` with 0.95 confidence, and gave the same result on repeated runs, as expected with low temperature.
- **ParallelAgent:** The searches genuinely overlap. All three started within 0.1 s of each other (t=0.1–0.2 s) and finished at 3.4, 3.9 and 4.9 s. The aggregator started only at t=4.9 s, after the last search completed (fan-out, then fan-in). Gathering takes about as long as the slowest search instead of the sum of all three. The aggregator kept the top 13 of 17 sources for the research stage.
- **LoopAgent:** The first draft scored 0.78 ("needs improvement"), so the critic didn't escalate and the `LoopAgent` started another iteration. The researcher's iteration notes confirm it used the critic's feedback from the session history. The revised draft scored 0.96, the critic escalated, and ADK ended the loop after 2 of 3 iterations.
- **Critic strictness matters:** With `gemini-3.5-flash-lite` as the critic, every draft scored 0.95 on the first pass, under both the default 0.80 rule and higher thresholds. Stronger critic models (`gemini-3.5-flash`, `gemini-3.8-flash`) scored the same draft 0.88–0.92. So the critic runs on `gemini-3.5-flash` while the other agents stay on the faster `flash-lite`.
- **Grounding helps:** Passing the Stage 2 sources into the loop produced strong first drafts. The final credibility score was 0.94.
- **Token limits:** Newer "thinking" Gemini models spend part of `max_output_tokens` on reasoning. This cut the JSON responses short and silently produced 0 sources. The fix was a non-thinking main model, plus higher limits: 2048 for the search agents, 3072 for citations, and 4096 for the critic.
- **Performance:** The whole run took about 37 s, and the evaluator rated it "excellent". That is roughly 11 s faster than the earlier version, which looked parallel but ran the searches one after another. Using a thinking model as the critic adds time compared with all-flash-lite.
