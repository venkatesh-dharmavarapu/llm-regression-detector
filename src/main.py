import json
import yaml
import ollama
from models import ClassificationOutput, PromptConfig

def load_prompt_config(filepath: str) -> PromptConfig:
    """Loads the YAML prompt versioning configuration."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return PromptConfig(**data)

def classify_email(email_text: str, config: PromptConfig) -> ClassificationOutput:
    """Sends the email text to the local Ollama model and parses the structured response."""
    user_content = f"Customer Email:\n\"\"\"\n{email_text}\n\"\"\""
    
    response = ollama.chat(
        model=config.model_name,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_content}
        ],
        options={"temperature": 0.0}  # Keep it deterministic for consistent evaluation runs
    )
    
    raw_output = response['message']['content'].strip()
    
    # Clean out accidental markdown formatting blocks if the model includes them
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
        
    try:
        json_data = json.loads(raw_output)
        return ClassificationOutput(**json_data)
    except Exception as e:
        # Fallback mechanism if the local model formatting fails
        print(f"Failed to parse model output: {raw_output}")
        raise e

if __name__ == "__main__":
    # Test execution block to verify functionality locally
    print("Loading prompt configuration...")
    config = load_prompt_config("prompts/email_classifier_v1.yaml")
    
    test_email = "Hey team, my credit card was charged twice for last month's subscription. Can I get a refund?"
    print(f"\nTesting email classification with model: {config.model_name}...")
    
    try:
        result = classify_email(test_email, config)
        print("\n--- Model Output ---")
        print(f"Category: {result.category}")
        print(f"Summary:  {result.summary}")
    except Exception as err:
        print(f"An error occurred: {err}")