from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


human_template = """#zh-tw The following is a friendly conversation between a human and an AI Assistent. 
The Assistent is talkative and provides lots of specific details from its context. 
If the Assistent does not know the answer to a question, it truthfully says it does not know.
        
Chat History:
{chat_history}
        
        
Current conversation:
Human:{question} 
Assistent:
"""

human_message_template = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful AI assistant built by ITRI."),
        human_message_template
    ]
)
