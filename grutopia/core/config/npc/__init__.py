from typing import Optional

from pydantic import BaseModel


class NPCUserConfig(BaseModel):
    name: str
    llm_model_name: str
    openai_api_key: str
    scene_data_path: str
    max_interaction_turn: int = 5
    api_base_url: Optional[str] = 'https://api.openai.com/v1/chat/completions'
