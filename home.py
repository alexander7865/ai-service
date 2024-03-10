import streamlit as st
VIDEO_URL = "https://youtu.be/zKGeRWjJlTU"
def app():
    st.snow()
    st.subheader(":coffee: 인공지능 가지고 놀기")
    st.divider()
    st.markdown("나는 **FLUTTER** 개발자다. 그런데 요즘 인공지능에 대해 관심이 많아 유튜브나 오픈AI 홈페이지를 참고해서 **인공지능 챗봇**(GPT-3.5-TURBO)을 만들었다.\n그런데 문제가 발생했다. 그 이유는 바로~~ 2021년까지 학습한 정보안에서만 답을 준다능...빡대가리 쉐키 ㅋㅋㅋ 해결방법은 GPT-4 를 이용하면 되지만 토큰비용이 ㄷㄷ ㅠㅠ\n그렇다고 인공지능을 활용하지 말아야 할까? 나는 그래서 다른방법으로 인공지능을 활용하기로 했다.\n인공지능에게 역할을 부여하는 것이다. 예를 들어 인공지능에게 **'너는 글을 잘 작성하는 블로거 이다'** 라고 하면 그 분야에 대해서 대답을 아주 잘해준다.ㅎㅎㅎ 이제부터 오픈AI를 활용해서 여러가지 프로젝트를 만들어 봐야겠다!")
    st.video(VIDEO_URL)  
    st.divider()
    
