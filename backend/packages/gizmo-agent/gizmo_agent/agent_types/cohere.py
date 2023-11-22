import os

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatCohere
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function


def get_cohere_function_agent(
    tools, system_message
):
    llm = ChatCohere()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    if tools:
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )
    else:
        llm_with_tools = llm
    agent = prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
    return agent
