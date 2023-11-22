from langchain.chat_models import ChatCohere
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools.render import render_text_description

from .prompts import conversational_prompt, parse_output


def _collapse_messages(messages):
    log = ""
    if isinstance(messages[-1], AIMessage):
        scratchpad = messages[:-1]
        final = messages[-1]
    else:
        scratchpad = messages
        final = None
    if len(scratchpad) % 2 != 0:
        raise ValueError("Unexpected")
    for i in range(0, len(scratchpad), 2):
        action = messages[i]
        observation = messages[i + 1]
        log += f"{action.content}<observation>{observation.content}</observation>"
    if final is not None:
        log += final.content
    return AIMessage(content=log)


def construct_chat_history(messages):
    collapsed_messages = []
    temp_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if temp_messages:
                collapsed_messages.append(_collapse_messages(temp_messages))
                temp_messages = []
            collapsed_messages.append(message)
        else:
            temp_messages.append(message)

    # Don't forget to add the last non-human message if it exists
    if temp_messages:
        collapsed_messages.append(_collapse_messages(temp_messages))

    return collapsed_messages


def get_cohere_xml_agent(tools, system_message):
    model = ChatCohere()

    prompt = conversational_prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
        system_message=system_message,
    )
    llm_with_stop = model.bind(stop=["</tool_input>"])

    agent = (
        {"messages": lambda x: construct_chat_history(x["messages"])}
        | prompt
        | llm_with_stop
        | parse_output
    )
    return agent
