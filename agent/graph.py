from langgraph.graph import StateGraph
from .state import AgentState
from .prompts import SYSTEM_PROMPT
from openai import OpenAI

client = OpenAI()

def generate_answer(state):

    context = "\n\n".join(state["context"])

    prompt = SYSTEM_PROMPT.format(context=context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":prompt},
            {"role":"user","content":state["user_message"]}
        ]
    )

    return {"response": response.choices[0].message.content}


graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("retrieve")

graph.add_edge("retrieve", "generate")

agent = graph.compile()