import os

from openai import AzureOpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class AzureOpenAIEngine:
    def __init__(self, deployment_name="gpt-4o"):
        self.deployment_name = deployment_name
        self.client = AzureOpenAI(
            azure_endpoint=os.environ.get("API_BASE"),
            azure_deployment=deployment_name,
            api_key=os.environ.get("GRAPHRAG_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
        )
        return response.choices[0].message.content