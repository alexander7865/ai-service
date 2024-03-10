import streamlit as st
import tiktoken
import os
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.faiss import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def app():

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 여기서 부터는 챗팅스크린 로직임
    with st.expander(label = ':books: **"문서"** & :key: **"OPENAI API_KEY"** 업로드'):
        uploaded_files =  st.file_uploader("분석할 문서를 업로드 하세요.",type=['pdf','docx','pptx'],accept_multiple_files=True)
        openai_api_key = st.text_input("당신의 OPENAI API_KEY 를 넣어주세요.", key="chatbot_api_key", type="password")
        st.markdown("[OPENAI API-KEY 받으러 가기](https://platform.openai.com/account/api-keys)",help="비용 문제로 인하여 해당 페이지에서 직접 API_KEY 를 받아서 사용하셔야 합니다.")
        process = st.button("문서분석")
    if process:
        if not openai_api_key:
            st.info("OPENAI API_KEY 값은 필수입니다.")
            st.stop()
        elif not uploaded_files:
            st.info("파일은 적어도 한개 이상 업로드 해주세요.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key)
        st.session_state.processComplete = True
    
    # chat title logic
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                        "content": "첨부한 문서에 대해 궁금하신 것이 있으면 언제든 질문해주세요!"}]
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    history = StreamlitChatMessageHistory(key="chat_messages")
    # chat body logic
    if not st.session_state.conversation:
        input_info = True
    else :
        input_info = False
    if query := st.chat_input("질문을 입력해주세요.",disabled=input_info): # disabled 입력상태가 전부 들어온다면 False 로 변경
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    # index 3 개를 하면 문제 발생
    
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
            
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo-16k-0613',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain.invoke
