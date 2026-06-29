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