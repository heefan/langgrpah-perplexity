"""
This node is responsible for creating the steps for the research process.
"""

from typing import List
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from copilotkit.langgraph import copilotkit_customize_config
from pydantic import BaseModel, Field
from ai_researcher.state import AgentState
from ai_researcher.utils import get_model
from langchain_core.messages import BaseMessage

class BreakdownStep(BaseModel):
    """Model for a search step"""

    id: str = Field(description="The id of the step. This is used to identify the step in the state. Just make sure it is unique.")
    description: str = Field(description='The description of the step, i.e. "search for information about the latest AI news"')
    status: str = Field(description='The status of the step. Always "pending".', enum=['pending'])
    type: str = Field(description='The type of step.', enum=['search'])


@tool
def BreakdownTool(steps: List[BreakdownStep]): 
    """
    Break the user's query into smaller steps.
    Use step type "search" to search the web for information.
    Make sure to add all the steps needed to answer the user's query.
    """


async def breakdown_node(state: AgentState, config: RunnableConfig):
    """
    The steps node is responsible for building the steps in the research process.
    """

    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[
            {
                "state_key": "steps",
                "tool": "SearchTool",
                "tool_argument": "steps"
            },
        ]
    )

    instructions = f"""
You are a search assistant. Your task is to help the user with complex search queries by breaking down into smaller steps.
These steps are then executed serially. In the end, a final answer is produced in markdown format.
The current date is {datetime.now().strftime("%Y-%m-%d")}.
"""

    response: BaseMessage = await get_model(state).bind_tools(
        [BreakdownTool],
        tool_choice="BreakdownTool"
    ).ainvoke([
        state["messages"][0],
        HumanMessage(
            content=instructions
        ),
    ], config)
    
    print("--------------------------------")
    print(response)
    print("----------END-------------------")

    if len(response.tool_calls) == 0:
        steps = []
    else:
        steps = response.tool_calls[0]["args"]["steps"]

    if len(steps) != 0:
        steps[0]["updates"] = ["Searching the web..."]

    return {
        "steps": steps,
    }
