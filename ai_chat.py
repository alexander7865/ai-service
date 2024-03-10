import streamlit as st
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import time
from dotenv import load_dotenv
load_dotenv()

def app():
    
    st.subheader("💬 인공지능 챗봇 (GPT-3.5-TURBO)")
    st.write("궁금한 내용을 물어보세요 친절하게 답해드릴께요")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-16k-0613"

    if "message" not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    # 유저의 input
    if prompt := st.chat_input("무엇이 궁금한가요?"):
        # 유저가 보낸 내용은 내용 그대로 사람 아이콘과 함께 화면에 출력하기 
        st.session_state.message.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # AI가 답변할 내용: 유저 인풋을 LLM에게 보내서 실행시키고, 받은 답변 실시간으로 출력하기 
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # 여기서부턴 위에서 만든 백엔드 코드가 들어갑니다! 
            # LLM
            llm = ChatOpenAI()
            
            # Prompt
            custom_prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        "You are a nice and smart chatbot having a conversation with a human."
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}"),
                ]
            )

            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            conversation = LLMChain(llm=llm, prompt=custom_prompt, verbose=True, memory=memory)
            
            # 여기서 {"question": prompt}는 유저의 인풋 프롬프트를 LLM에게 보낸다는 뜻
            result = conversation.invoke({"question": prompt})
            # 실시간으로 텍스트가 로딩되는 효과를 보여주기 위해 결과값을 단어 단위로 쪼개주기
            for chunk in result['text'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.message.append({"role": "assistant", "content": full_response})