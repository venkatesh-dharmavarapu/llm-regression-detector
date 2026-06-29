# 🛡️ Automated LLM Regression Detection System

A production-grade continuous evaluation and quality-assurance pipeline designed to capture performance degradation, formatting drift, and classification regressions in LLM-powered features. This framework tests changes against a human-verified golden dataset and alerts teams via Slack before bad prompt completions reach users.

## 🗺️ System Architecture & Data Flow

[Golden Dataset] ---> [Async Test Runner] ---> [Local Ollama Instance]
|                      |
v                      v
[Diff Engine] <--- [Pydantic Validation (Strict Contract)]
|
+--------------------+--------------------+
|                                         |
v                                         v
[Slack Webhook Alerts]                  [Streamlit Dashboards]

1. **Prompt Configuration:** Prompts are isolated from application code and versioned via structured YAML files within the `/prompts` directory.
2. **Interface Contract:** Customer emails are processed through a local `qwen2.5:3b` engine and structurally validated using strict Pydantic parsing.
3. **Multi-Dimensional Evaluation:** The system computes hard accuracy constraints alongside an LLM-as-a-Judge semantic quality matrix driven by a local `llama3:8b` model.
4. **Regression Diffing:** Current executions are automatically analyzed against persistent baseline run logs (`data/latest_run.json`).
5. **Proactive Alerting:** Diffs exceeding performance boundaries immediately dispatch markdown payload blocks to an engineering Slack channel endpoint.

---

## 🛠️ Local Development Setup

### Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) running locally with both `qwen2.5:3b` and `llama3:8b` models pulled:
  ```bash
  ollama pull qwen2.5:3b
  ollama pull llama3:8b

Installation & Initialization
Clone the repository and initialize the isolated environment:

python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

Configure local environment properties by creating a .env file at the root:

Code snippet
SLACK_WEBHOOK_URL="[https://hooks.slack.com/services/YOUR/WEBHOOK/LINK](https://hooks.slack.com/services/YOUR/WEBHOOK/LINK)"
🚀 Execution & Operational Workflows
1. Run the Evaluation Suite
To execute the automated evaluation matrix, calculate data drift, check for regressions, and fire Slack alerts:

Bash
python src/evaluator.py
2. Launch the Performance Scorecard Dashboard
To view interactive trend analysis graphs, global latency reports, and side-by-side completion text diffs:

Bash
streamlit run src/pipeline.py

📈 Production Insights & Design Decisions

Why Hand-Curation over Synthetic Generation?
An evaluation engine is bounded by the quality of its underlying verification data. Rather than using an LLM to generate generic test samples, the golden dataset was deliberately seeded with hand-crafted edge cases, mixed-language text, orthographic errors (typos), and complex multi-intent requests. This ensures our baseline reflects high-friction human interactions.

The Double-Edged Prompt Dilemma (Real-World Case Study)
During testing, a prompt optimization introduced to resolve an edge case involving account cancellations successfully fixed that specific bug, but instantly degraded categorical accuracy across two previously healthy segments. Without this continuous integration framework, that regression would have broken production workflows undetected—proving the absolute necessity of regression pipelines in AI engineering.