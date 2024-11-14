import streamlit as st
from streamlit import session_state as ss
import asyncio
from yeah import MyThreadPoolExecutor
from tempfile import TemporaryFile

from chatbot import seterr, setsta, depos_setup, talk, separator

for e in ["disabled", "load", "running", "loaderror", "setup", "end"]:
    if e not in ss:
        ss[e] = False

if "address" not in ss:
    ss["address"] = ""

if "summary_kor" not in ss:
    ss["summary_kor"] = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if ss["loaderror"]:
    ss["loaderror"] = False
    ss["running"] = False
    ss["load"] = False
    ss.lo = False
    st.error("불러오는 중 오류가 발생하였습니다.")

if not ss["running"]:
    st.title("R&E 리뷰 분석 이끾끼 프로그램")
    deletion = st.empty()
    setup_cont = deletion.container()
    input_cont = setup_cont.container()
    lo = setup_cont.checkbox("파일에서 불러오기",key="lo")
    if lo:
        ss["disabled"] = True
        ss["address"] = ""
    else:
        ss["disabled"] = False
    ad = input_cont.text_input("주소 입력",key="addr",disabled=ss["disabled"],value=ss["address"])
    ss["address"] = ad
    ss["load"] = lo
    uploaded_file = None
    if lo:
        uploaded_file = setup_cont.file_uploader("불러올 파일 업로드",type="cyc1",accept_multiple_files=False)
    star = setup_cont.button("시작")
    if star:
        deletion.empty()
        ss["running"] = True
        if lo:
            ss["uploaded"] = uploaded_file
        st.rerun()
elif not ss["end"]:
    if not ss["setup"]:
        why_does_streamlit_display_thinking_when_langchain_starts_come_on = st.container(height=0,border=False)
        setsta(st.status("준비 중",expanded=True))
        seterr(st.container())
        asyncio.get_event_loop_policy().set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        try:
            with why_does_streamlit_display_thinking_when_langchain_starts_come_on:
                loop.set_default_executor(MyThreadPoolExecutor(max_workers=2))
                loop.run_until_complete(depos_setup(ss["load"], ss["address"]))
        except Exception as e:
            st.write(e)
            pass
        finally:
            loop.close()
        st.rerun()
    else:
        st.status(label="준비 완료", state="complete")
        st.expander(label="리뷰 요약 보기").write(ss["summary_kor"])
    
    st.divider()

    for message in ss["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    disable = False
    if "ques" in ss and ss.ques:
        disable = True
    st.chat_input("질문을 입력하세요",key="ques",disabled=disable)
    if "ques" in ss and ss.ques:
        prompt = ss.ques
        if prompt.find("^_^") != -1:
            ss["end"] = True
            st.divider()
            st.container(border=True).text("대화가 종료되었습니다. 다음에 사용하기 위해 수집한 리뷰들을 저장하려면 저장 버튼을 눌러 주세요.")
            file = TemporaryFile("w+b")
            try:
                file.write(bytes(ss['setup_dict']['summary'], encoding="utf-8"))
                file.write(separator)
                file.write(ss['setup_dict']['vectorstore'].serialize_to_bytes())
                file.seek(0)
                data = file.read()
            except:
                st.error("오류가 발생하였습니다. 수집한 리뷰를 저장할 수 없습니다.")
                st.button("재시작")
            else:
                c1, c2 = st.columns([1,9])
                c1.download_button("저장", data, file_name="reviews.cyc1")
                c2.button("재시작")
            st.stop()
        ss["messages"].append({"role":"user", "content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = asyncio.run(talk(ss["setup_dict"], prompt))
        ss["messages"].append({"role":"assistant", "content":response})
        with st.chat_message("assistant"):
            st.markdown(response)
        st.rerun()
else:
    ss.clear()
    st.rerun()