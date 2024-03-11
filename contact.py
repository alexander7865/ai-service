import streamlit as st



def app():
    st.subheader(":email: 이메일 문의")

    contact_form = """
    <form action="https://formsubmit.co/keyplayer0314@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="이름" required>
        <input type="email" name="email" placeholder="이메일" required>
        <textarea name="message" placeholder="메세지를 입력하세요"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html= True)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

    local_css("style.css")
