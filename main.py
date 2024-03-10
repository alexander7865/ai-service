import streamlit as st

from streamlit_option_menu import option_menu

import home, ai_chat, ai_analyze, contact

st.set_page_config(
    page_title="인공지능 헬퍼",
    page_icon=":brain:",
    layout="wide"
    )

class Multiapp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title":title,
            "function":function
        })

    def run():
        with st.sidebar:
            app = option_menu(
                menu_title='인공지능 서비스',
                options=['HOME','인공지능 챗봇','인공지능 문서분석 Q&A 챗봇','CONTACT'],
                icons=['house','bi-robot','book','envelope'],
                menu_icon="bi-cpu",
                default_index=0,
                styles={
                    "container": {"padding": "5!important","background-color":'white'},
                    "nav-link": {"color":"black","font-size": "15px", "text-align": "left", "margin":"0px"},
                    "nav-link-selected": {"color":"white","font-size": "15px","background-color": "pink"}
                    })

        if app == "HOME":
            home.app()
        if app == "인공지능 챗봇":
            ai_chat.app()
        if app == "인공지능 문서분석 Q&A 챗봇":
            ai_analyze.app()
        if app == "CONTACT":
            contact.app()

             
    run()  