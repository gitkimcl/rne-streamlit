import streamlit as st
from streamlit import session_state as ss
import asyncio
from yeah import MyThreadPoolExecutor

from chatbot import seterr, setsta, depos_setup

if "disabled" not in ss:
    ss["disabled"] = False

if "address" not in ss:
    ss["address"] = ""

if "load" not in ss:
    ss["load"] = False

if "running" not in ss:
    ss["running"] = False

if "setup" not in ss:
    ss["setup"] = False

if "loaderror" not in ss:
    ss["loaderror"] = False

if ss.loaderror:
    ss["loaderror"] = False
    ss["running"] = False
    ss["load"] = False
    ss.lo = False
    st.error("불러오는 중 오류가 발생하였습니다.")

if not ss.setup:
    if not ss.running:
        deletion = st.empty()
        setup_cont = deletion.container()
        input_cont = setup_cont.container()
        lo = setup_cont.checkbox("파일에서 불러오기",key="lo")
        if lo:
            ss["disabled"] = True
            ss["address"] = ""
        else:
            ss["disabled"] = False
        ad = input_cont.text_input("주소 입력",key="addr",disabled=ss.disabled,value=ss.address)
        ss["address"] = ad
        ss["load"] = lo
        star = setup_cont.button("시작")
        if star:
            deletion.empty()
            ss["running"] = True
            st.rerun()
    else:
        why_does_streamlit_display_thinking_when_langchain_starts_come_on = st.container(height=0,border=False)
        seterr(st.container())
        setsta(st.status("준비 중",expanded=True))
        seterr(st.container())
        try:
            with why_does_streamlit_display_thinking_when_langchain_starts_come_on:
                asyncio.get_event_loop_policy().set_event_loop(asyncio.new_event_loop())
                asyncio.get_event_loop().set_default_executor(MyThreadPoolExecutor(max_workers=2))
                asyncio.get_event_loop().run_until_complete(depos_setup(ss.load, ss.address))
        except Exception as e:
            st.write(e)
            pass
        ss.clear()
else:
    st.container(border=True).text(ss.setup_dict['summary'])