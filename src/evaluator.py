import json
import time
import os
from typing import List
import ollama
import httpx  # Ensure this is imported at the top
from models import TestCase, EvalResult
from main import load_prompt_config, classify_email

JUDGE_MODEL = "llama3:8b"
RESULTS_FILE = "data/latest_run.json"
# Place your Slack Incoming Webhook URL here (or leave empty to skip network calls)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

def load_golden_dataset(filepath: str) -> List[TestCase]:
    with open(filepath, 'r') as file:
        data = json.load(file)
    return [TestCase(**item) for item in data]

def judge_summary(email_text: str, expected_summary: str, actual_summary: str) -> int:
    judge_prompt = f"""
    You are an impartial quality control evaluator. Your job is to score a generated summary of a customer support email.
    Original Email: "{email_text}"
    Ground Truth Summary: "{expected_summary}"
    Generated Summary: "{actual_summary}"
    Respond with EXACTLY one integer between 1 and 5. No filler.
    """
    try:
        response = ollama.chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": judge_prompt}], options={"temperature": 0.0})
        score_text = response['message']['content'].strip()
        score = int(''.join(filter(str.isdigit, score_text))[0])
        return min(max(score, 1), 5)
    except Exception:
        return 3

def send_slack_alert(accuracy: float, failed_count: int, regressions: list):
    """Sends a structured alert block to your Slack channel via incoming webhooks."""
    if not SLACK_WEBHOOK_URL:
        print("\nℹ️ Slack webhook URL not set. Skipping notification.")
        return

    status_icon = "🚨 CRITICAL FAILURE" if failed_count > 0 else "✅ SYSTEM HEALTHY"
    
    payload = {
        "text": f"LLM Regression Test Run Result: {status_icon}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🛡️ Model Evaluation Report"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Status:*\n{status_icon}"},
                    {"type": "mrkdwn", "text": f"*Categorical Accuracy:*\n{accuracy:.2f}%"}
                ]
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Failures:*\n{failed_count} cases"},
                    {"type": "mrkdwn", "text": f"*Regressed IDs:*\n{', '.join(regressions) if regressions else 'None'}"}
                ]
            }
        ]
    }
    
    try:
        response = httpx.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print("📢 Slack regression notification dispatched successfully!")
    except Exception as e:
        print(f"Failed to send Slack alert: {e}")

def run_evaluation(dataset_path: str, prompt_config_path: str) -> List[EvalResult]:
    config = load_prompt_config(prompt_config_path)
    test_cases = load_golden_dataset(dataset_path)
    results = []
    print(f"Starting Evaluation Run using Prompt Version: {config.version}\n")
    for case in test_cases:
        start_time = time.time()
        try:
            prediction = classify_email(case.input, config)
            latency = round(time.time() - start_time, 3)
            cat_match = prediction.category == case.expected_category
            score = judge_summary(case.input, case.expected_summary, prediction.summary)
            
            results.append(EvalResult(
                test_case_id=case.id, input_text=case.input,
                expected_category=case.expected_category.value, actual_category=prediction.category.value,
                category_match=cat_match, expected_summary=case.expected_summary,
                actual_summary=prediction.summary, summary_judge_score=score, latency_seconds=latency
            ))
            print(f"Processed {case.id} | Match: {cat_match}")
        except Exception as e:
            print(f"Error processing {case.id}: {e}")
    return results

def perform_diff_analysis(current_results: List[EvalResult]):
    total = len(current_results)
    passed_cat = sum(1 for r in current_results if r.category_match)
    failed_cat = total - passed_cat
    accuracy = (passed_cat / total) * 100 if total > 0 else 0

    regressions = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as file:
            historical_data = json.load(file)
        baseline_lookup = {res['test_case_id']: res for res in historical_data}
        for current in current_results:
            baseline = baseline_lookup.get(current.test_case_id)
            if baseline and baseline['category_match'] and not current.category_match:
                regressions.append(current.test_case_id)

    print("\n" + "="*25 + " REGRESSION ANALYSIS " + "="*25)
    print(f"📉 Regressions detected: {len(regressions)} {regressions}")
    print("=" * 71)
    
    # Trigger the notification dispatch loop
    send_slack_alert(accuracy, failed_cat, regressions)

if __name__ == "__main__":
    DATASET = "data/golden_dataset.json"
    PROMPT_CONFIG = "prompts/email_classifier_v1.yaml"
    
    run_results = run_evaluation(DATASET, PROMPT_CONFIG)
    perform_diff_analysis(run_results)
    
    with open(RESULTS_FILE, 'w') as file:
        json.dump([res.model_dump() for res in run_results], file, indent=2)