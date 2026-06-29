from enum import Enum
from pydantic import BaseModel, Field

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"

class ClassificationOutput(BaseModel):
    category: EmailCategory = Field(description="The primary category of the customer support email.")
    summary: str = Field(description="A concise, one-sentence summary of the email.")

class PromptConfig(BaseModel):
    version: str
    system_prompt: str
    model_name: str

class TestCase(BaseModel):
    id: str
    input: str
    expected_category: EmailCategory
    expected_summary: str
    difficulty: str
    notes: str

class EvalResult(BaseModel):
    test_case_id: str
    input_text: str
    expected_category: str
    actual_category: str
    category_match: bool
    expected_summary: str
    actual_summary: str
    summary_judge_score: int = 0  # We will populate this in Phase 3
    latency_seconds: float