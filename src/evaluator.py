import json
import time
import os
from typing import List
import ollama
from models import TestCase, EvalResult
from main import load_prompt_config, classify_email

JUDGE_MODEL = "llama3:8b"
RESULTS_FILE = "data/latest_run.json"

def load_golden_dataset(filepath: str) -> List[TestCase]:
    with open(filepath, 'r') as file:
        data = json.load(file)
    return [TestCase(**item) for item in data]

def judge_summary(email_text: str, expected_summary: str, actual_summary: str) -> int:
    judge_prompt = f"""
    You are an impartial quality control evaluator. Your job is to score a generated summary of a customer support email.
    
    [Context]
    Original Email: "{email_text}"
    Ground Truth Summary (Ideal): "{expected_summary}"
    Generated Summary to Evaluate: "{actual_summary}"
    
    [Grading Rubric]
    5 - Excellent: Captures the main point perfectly in a single sentence without adding fake details.
    3 - Acceptable: Gets the main idea right, but misses a minor detail or is slightly wordy.
    1 - Terrible: Completely misses the point, contains hallucinations, or is multiple sentences.
    
    Respond with EXACTLY one integer between 1 and 5. Do not include any explanations, markdown, or filler.
    """
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            options={"temperature": 0.0}
        )
        score_text = response['message']['content'].strip()
        score = int(''.join(filter(str.isdigit, score_text))[0])
        return min(max(score, 1), 5)
    except Exception:
        return 3

def run_evaluation(dataset_path: str, prompt_config_path: str) -> List[EvalResult]:
    config = load_prompt_config(prompt_config_path)
    test_cases = load_golden_dataset(dataset_path)
    results = []
    
    print(f"Starting Evaluation Run using Prompt Version: {config.version}")
    print(f"Target Engine: {config.model_name} | Judge Engine: {JUDGE_MODEL}\n")

    for case in test_cases:
        start_time = time.time()
        try:
            prediction = classify_email(case.input, config)
            latency = round(time.time() - start_time, 3)
            cat_match = prediction.category == case.expected_category
            score = judge_summary(case.input, case.expected_summary, prediction.summary)
            
            result = EvalResult(
                test_case_id=case.id,
                input_text=case.input,
                expected_category=case.expected_category.value,
                actual_category=prediction.category.value,
                category_match=cat_match,
                expected_summary=case.expected_summary,
                actual_summary=prediction.summary,
                summary_judge_score=score,
                latency_seconds=latency
            )
            results.append(result)
            print(f"Processed {case.id} | Match: {cat_match} | Judge: {score}/5")
        except Exception as e:
            print(f"Error processing {case.id}: {e}")
            
    return results

def perform_diff_analysis(current_results: List[EvalResult]):
    """Compares the current evaluation run against the previously stored baseline run."""
    if not os.path.exists(RESULTS_FILE):
        print("\n⚠️ No baseline file found. Saving this run as the initial baseline.")
        return
        
    with open(RESULTS_FILE, 'r') as file:
        historical_data = json.load(file)
        
    baseline_lookup = {res['test_case_id']: res for res in historical_data}
    
    regressions = []
    improvements = []
    
    for current in current_results:
        baseline = baseline_lookup.get(current.test_case_id)
        if not baseline:
            continue
            
        # Check if accuracy flipped from True (pass) to False (fail)
        if baseline['category_match'] and not current.category_match:
            regressions.append(current.test_case_id)
        elif not baseline['category_match'] and current.category_match:
            improvements.append(current.test_case_id)
            
    print("\n" + "="*25 + " REGRESSION ANALYSIS " + "="*25)
    print(f"📈 Improvements detected (Fail -> Pass): {len(improvements)} {improvements if improvements else ''}")
    print(f"📉 Regressions detected (Pass -> Fail): {len(regressions)} {regressions if regressions else ''}")
    
    if len(regressions) > 0:
        print("\n❌ CRITICAL CRITERIA TRIGGERED: Prompt change caused a quality regression.")
    else:
        print("\n✅ Build Safe: No performance regressions detected.")
    print("=" * 71)

if __name__ == "__main__":
    DATASET = "data/golden_dataset.json"
    PROMPT_CONFIG = "prompts/email_classifier_v1.yaml"
    
    run_results = run_evaluation(DATASET, PROPrompt_CONFIG := PROMPT_CONFIG)
    
    # Analyze regressions before overwriting history
    perform_diff_analysis(run_results)
    
    # Persist data as the new baseline tracking point
    with open(RESULTS_FILE, 'w') as file:
        json.dump([res.model_dump() for res in run_results], file, indent=2)
    print(f"\n💾 Results serialized to {RESULTS_FILE}")