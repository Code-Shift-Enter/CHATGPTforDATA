
import streamlit as st
from streamlit_chat import message
import os
import pexpect

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat2yourData", page_icon="📊")
st.header("Chat2yourData")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

#from OPEN_AI_KEY_SAVE import OPEN_AI_KEY
import os
OPEN_AI_KEY = st.text_input("Please enter your OpenAI API-KEY", value="", type="password")
os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY
#openai_api_key = OPEN_AI_KEY

from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from typing import Any, Dict, List

df = pd.DataFrame([])
data = st.file_uploader(label='Upload CSV file', type='csv')



header_num = st.number_input(label='Header position',value=0)
index_num = st.number_input(label='Index position',value=2)
index_list = [i for i in range(index_num)]

if data:
    df = pd.read_csv(data,header=header_num,index_col=index_list)
    st.dataframe(df)

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

ask_button = ""

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df,openai_api_key=OPEN_AI_KEY, verbose=True, return_intermediate_steps=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['English'])

import json
import re
from collections import namedtuple
AgentAction = namedtuple('AgentAction', ['tool', 'tool_input', 'log'])

def format_action(action, result):
    action_fields = '\n'.join([f"{field}: {getattr(action, field)}"+'\n' for field in action._fields])
    return f"{action_fields}\nResult: {result}\n"

if ask_button:
#     res_box = st.empty()
    with st.spinner('typing...'):
        prefix = f'You are the best explainer. please answer in {language}. User: '
        handler = SimpleStreamlitCallbackHandler()
        response = agent({"input":user_input}) #,"callbacks":handler})
        
        
        actions = response['intermediate_steps']
        actions_list = []
        for action, result in actions:
            text = f"""Tool: {action.tool}\n
               Input: {action.tool_input}\n
               Log: {action.log}\nResult: {result}\n
            """
            text = re.sub(r'`[^`]+`', '', text)
            actions_list.append(text)
            
        answer = json.dumps(response['output'],ensure_ascii=False).replace('"', '')
        if language == 'English':
            with st.expander('ℹ️ Show details', expanded=False):
                st.write('\n'.join(actions_list))

            
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
