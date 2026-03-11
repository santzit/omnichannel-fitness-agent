from typing import TypedDict, List

class AgentState(TypedDict):
    user_message: str
    context: List[str]
    response: str