import time
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from langchain.docstore.document import Document

from operator import itemgetter, add
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, List, TypedDict

from langchain_openai.embeddings import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

import streamlit as st
from streamlit import session_state as ss

error_place = None
sta = None
rpt = ChatOpenAI(temperature=0.3, model ="gpt-4o-mini",openai_api_key=st.secrets["openai-api-key"])
slice_size = 10
separator = b"""Hello. this is a separator. I don't know separator is the right spelling of separator. Anyways this separates things. If you somehow get this exact text in something like vectorstore, it will break the program. So don't try to intentionally put this text somewhere else. Ecyc e. HAhahahahahahahaaihiojiojaiao;hawio;heaoa;uelamafhxeufiaewnfaehwufieheuvfihvaeuigehguirahelrtahewuiwelathwueflhewaufnilhxuailmwe"""
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
options.add_argument("disable-dev-shm-usage")

def seterr(e):
    global error_place
    error_place = e

def setsta(s):
    global sta
    sta = s

class SetupOverallState(TypedDict):
    link: str
    reviews: List
    merged: List
    translated: List
    summaries: Annotated[List, add]
    final_summary: str
    vectorstore: FAISS

def get_link(link):
    return {"link": link}

def log_fetch_process(len: int):
    sta.update(label=f"리뷰 수집: {len}개 수집됨")

def fetch_11avenue(state: SetupOverallState):
    dr = webdriver.Chrome(service=Service(executable_path='/usr/bin/chromedriver'),options=options)
    dr.get(state['link'])

    ifrm = dr.find_element(By.ID, 'ifrmReview')
    dr.switch_to.frame(ifrm)

    texts = []

    while len(texts) < 1000:
        log_fetch_process(len(texts))
        try:
            WebDriverWait(dr, 3).until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".review-next-list")))
            dr.execute_script('window.scrollTo(0, document.querySelector(".review-next-list").getBoundingClientRect().y)')
            
            texts += dr.execute_script("""
                ret = [];
                while (true) {
                    a = document.querySelector(".review_list_element");
                    if (a == null) { break; }
                    b = a.querySelector(":scope .cont_review_hide");
                    if (b != null) { ret.push(b.textContent); }
                    a.remove();
                }
                return ret;
            """)

            WebDriverWait(dr, 3).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".review-next-list"))).click()
        except TimeoutException as e:
            break
        except Exception as e:
            raise e

    dr.quit()

    texts_2 = [e.strip(' \n').replace('\xa0', ' ').replace('\n', ' ') for e in filter(None,texts)]
    return {"reviews": texts_2}

def fetch_newworld(state: SetupOverallState):
    dr = webdriver.Chrome(service=Service(executable_path='/usr/bin/chromedriver'),options=options)
    dr.get(state['link'])

    texts_2 = []

    while len(texts_2) < 1000:
        log_fetch_process(len(texts_2))
        try:
            for i in range(1,11) :
                for j in range(1,11) :
                    WebDriverWait(dr, 2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, f'#item_rvw_list > li:nth-child({j})')))
                    try:
                        if dr.find_elements(By.CSS_SELECTOR, f'#item_rvw_list > li:nth-child({j}) .rvw_item_thumb > img').size() != 0:
                            WebDriverWait(dr, 2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, f'#item_rvw_list > li:nth-child({j}) .rvw_item_thumb > img')))
                    except:
                        pass
                    texts = dr.execute_script(f"""
                        a = document.querySelector("#item_rvw_list > li:nth-child({j}) > div.rvw_expansion_panel_head > a > div > div.rvw_panel_expand_hide_group > p")
                        return a.textContent
                    """)
                    texts_2.append(texts.strip(' \n').replace('\n', ' '))
                
                log_fetch_process(len(texts_2))
                
                if i == 10:
                    continue
                
                if len(dr.find_elements(By.CSS_SELECTOR, f'#comment_navi_area > div > a:nth-child({i+1})')) == 0:
                    break

                WebDriverWait(dr, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, f'#comment_navi_area > div > a:nth-child({i+1})')))
                dr.execute_script(f'window.scrollTo(0, document.querySelector("#comment_navi_area > div > a:nth-child({i+1})").getBoundingClientRect().y)')
                time.sleep(0.2)
                dr.execute_script(f'window.scrollBy(0, -40)')
                WebDriverWait(dr, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'#comment_navi_area > div > a:nth-child({i+1})'))).click()
            
            if len(dr.find_elements(By.CSS_SELECTOR, f'.rvw_btn_next')) == 0:
                break
        except Exception as e:
            break

        WebDriverWait(dr, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, f'.rvw_btn_next')))
        dr.execute_script(f'window.scrollTo(0, document.querySelector(".rvw_btn_next").getBoundingClientRect().y)')
        time.sleep(0.2)
        dr.execute_script(f'window.scrollBy(0, -40)')
        WebDriverWait(dr, 20).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f'.rvw_btn_next'))).click()

    dr.quit()
    return {"reviews": texts_2}

def fetch_nogent(state: SetupOverallState):
    dr = webdriver.Chrome(service=Service(executable_path='/usr/bin/chromedriver'),options=options)

    new_link = state['link'].replace("products", "review/goods")
    dr.get(new_link)

    j=0
    texts_2 = []
    end_flag = 0

    try:
        while True:
            log_fetch_process(len(texts_2))
            try:
                texts = dr.execute_script(f"""
                    a = document.querySelector("div[data-index=\\"{j}\\"] p");
                    t = a.textContent;
                    return t;
                """)
                texts_2.append(texts.strip(' \n'))
                j += 1
                end_flag = 0
            except KeyboardInterrupt:
                break
            except:
                dr.execute_script("window.scrollBy(0, 450)")
                WebDriverWait(dr, 2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, f'div[data-index="{j-1}"] p')))
                end_flag += 1
                if end_flag == 100:
                    break
                continue

            if len(texts_2) >= 1000:
                break
    except Exception:
        pass

    dr.quit()
    return {"reviews": texts_2}

def choose_fetch(state: SetupOverallState) -> Literal["get_11avenue", "get_newworld", "get_nogent"]:
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 수집 시작")
    sta.update(label=f"리뷰 수집 시작")
    try:
        if "11st.co.kr" in state['link']:
            return "fetch_11avenue"
        elif "ssg.com" in state['link']:
            return "fetch_newworld"
        elif "musinsa.com" in state['link']:
            return "fetch_nogent"
        else:
            raise AssertionError("잘못된 주소거나 지원하지 않는 쇼핑몰의 주소입니다")
    except Exception as e:
        raise e

def end_fetch(state: SetupOverallState):
    log_fetch_process(len(state['reviews']))
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 수집 완료({len(state['reviews'])}개)")
    sta.update(label=f"리뷰 수집 완료")
    return None

def merge_reviews(state: SetupOverallState):
    texts_sliced = []
    for i in range(slice_size):
        texts_sliced.append(state['reviews'][i::slice_size])

    for i in range(slice_size):
        texts_sliced[i] += [""] * (len(texts_sliced[0]) - len(texts_sliced[i]))

    texts_zip = zip(*texts_sliced)

    merged_text = []
    for t in texts_zip:
        merged_text.append("\n--------\n".join(t))

    return {"merged": merged_text}

translate_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Your task is to translate a given text into a specified language while preserving the
    original tone of the text. Note that the original text contains a pattern to split reviews, and you have to preserve it.
    Target language: english
    Split pattern: ```\n--------\n```"""),
    ("human", "{string}")
])

Maus = translate_prompt_template | rpt | StrOutputParser()

def translate(state: SetupOverallState):
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 번역 시작")
    sta.update(label=f"리뷰 번역 시작")
    result = Maus.batch(state['merged'])
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 번역 완료")
    sta.update(label=f"리뷰 번역 완료")
    return {"translated": result}
map_prompt = ChatPromptTemplate([("human", """You will be given up to 10 reviews from different customer. Summerize them, ideally in less than 80 words.
    If the given reviews contain different opinions, contain all of them in the summary.
    
    {context}""")])

reduce_prompt = ChatPromptTemplate([("human", """Combine these summaries. Try to include every opinions.
    
    {context}""")])

map_chain = map_prompt | rpt | StrOutputParser()
reduce_chain = reduce_prompt | rpt | StrOutputParser()

class SummaryState(TypedDict):
    content: str

async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state['content'])
    return {"summaries": [response]}

def map_summaries(state: SetupOverallState):
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 요약 시작")
    sta.update(label=f"임베딩 및 리뷰 요약 시작")
    return [
        Send("generate_summary", {"content": content}) for content in state['translated']
    ]

async def generate_final_summary(state: SetupOverallState):
    response = await reduce_chain.ainvoke(state['summaries'])
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 리뷰 요약 완료")
    sta.update(label=f"리뷰 요약 완료")
    return {"final_summary": response}
embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai-api-key"])
index = faiss.IndexFlatL2(len(embeddings.embed_query("ecyc e")))

def embed(state: SetupOverallState):
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 임베딩 시작")
    # 가독성 버려!!!!
    split = [ee for e in [e.split("\n--------\n") for e in state['translated']] for ee in e]
    doc = [Document(page_content=e) for e in split]

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(documents=doc)
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 임베딩 완료")
    sta.update(label=f"임베딩 완료")
    return {"vectorstore": vector_store}

setup_graph = StateGraph(SetupOverallState)

setup_graph.add_node("fetch_11avenue", fetch_11avenue)
setup_graph.add_node("fetch_newworld", fetch_newworld)
setup_graph.add_node("fetch_nogent", fetch_nogent)
setup_graph.add_node("end_fetch", end_fetch)
setup_graph.add_conditional_edges(START, choose_fetch, ["fetch_11avenue", "fetch_newworld", "fetch_nogent"])
setup_graph.add_edge("fetch_11avenue", "end_fetch")
setup_graph.add_edge("fetch_newworld", "end_fetch")
setup_graph.add_edge("fetch_nogent", "end_fetch")
# setup_graph.add_edge("end_fetch", END)

setup_graph.add_node("merge_reviews", merge_reviews)
setup_graph.add_node("translate", translate)
setup_graph.add_node("generate_summary", generate_summary)
setup_graph.add_node("generate_final_summary", generate_final_summary)
setup_graph.add_edge("end_fetch", "merge_reviews")
setup_graph.add_edge("merge_reviews", "translate")
setup_graph.add_conditional_edges("translate", map_summaries, ["generate_summary"])
setup_graph.add_edge("generate_summary", "generate_final_summary")
# setup_graph.add_edge("generate_final_summary", END)

setup_graph.add_node("embed", embed)
setup_graph.add_edge("translate", "embed")

setup_graph.add_edge("generate_final_summary", END)
setup_graph.add_edge("embed", END)

setup_comp = setup_graph.compile()

setup_chain = setup_comp | {"summary": itemgetter("final_summary"), "vectorstore": itemgetter("vectorstore")}

# setup = await setup_chain.ainvoke(get_link())
# setup
chatbot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI assistant that provides answers about a product based on customer reviews.
        You have access to the summary of reviews and a small set of reviews related to the question.
        When the user asks a question babout the product, answer based on the reviews is provided.
        If the information is not available in the reviews, answer the question nonetheless, but also say that the question was unrelated to the reviews.
        Please refrain from saying that all the reviews say the same thing, as you will be given only the fraction of all reviews. Instead, say that all the reviews you found said the same.
        Keep your responses concise, focused on the question, and ensure they are grounded in the feedback from the reviews. 
        --- Summary ---
        {summary}
        --- Reviews relevant to the question ---
        {string}"""
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{messages}")
])

memory = {}  # memory is maintained outside the chain

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in memory:
        memory[session_id] = InMemoryChatMessageHistory()
    return memory[session_id]


_mrpt = RunnableWithMessageHistory(
    chatbot_prompt | rpt,
    get_session_history=get_session_history,
    input_messages_key="messages",
    history_messages_key="history"
)

mrpt = RunnableLambda(lambda question : _mrpt.invoke(question,config={"configurable":{"session_id":"ecyc_e"}}))
def call_app(input: dict):
    config={"configurable": {"thread_id": "abc456"}}
    input_messages = [HumanMessage(input['query'])]
    string=input['vectorstore'].similarity_search(query=input['query'], k = 8)
    output = mrpt.invoke(
        {"messages": input_messages, "string": string, "summary": input['summary']},
        config,
    )
    return output

async def talk(setup: dict):
    while True:
        question = st.chat_input("질문을 입력하세요")
        if question == "":
            print("\x1b[31m<(종료)\x1b[39m")
            break
        print(f"\x1b[33m<\x1b[39m {question}")
        out = await chain.ainvoke({"summary": setup['summary'],"vectorstore": setup['vectorstore'],"query": question})
        print(f"\x1b[34m>\x1b[39m {out.content}",flush=True)

chain = RunnableLambda(call_app)

async def depos_setup(loadfile, link): # 디포즈 셋업 해주시죠
    if loadfile:
        setup = {}
        try:
            with open("setup.dat", "rb") as f:
                load = f.read().split(separator)
                setup['summary'] = str(load[0], encoding="utf-8")
                setup['vectorstore'] = FAISS.deserialize_from_bytes(load[1], embeddings=embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"\x1b[31m{repr(e)}\x1b[39m")
            ss["loaderror"] = True
            st.rerun()
    else:
        try:
            setup = await setup_chain.ainvoke({"link":link})
        except Exception as e:
            sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 오류 발생: {repr(e)}")
            sta.update(label=f"오류 발생", state="error")
            print(f"\x1b[31m{repr(e)}\x1b[39m")
            ss.running = False
            error_place.button("재시작")
            st.stop()
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 준비 완료")
    sta.update(label=f"준비 완료", state="complete", expanded=False)
    ss["setup_dict"] = setup
    ss["setup"] = True
    error_place.container(border=True).text(ss.setup_dict['summary'])

"""async def chatbot(loadfile, link):
    if loadfile:
        setup = {}
        try:
            with open("setup.dat", "rb") as f:
                load = f.read().split(separator)
                setup['summary'] = str(load[0], encoding="utf-8")
                setup['vectorstore'] = FAISS.deserialize_from_bytes(load[1], embeddings=embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"\x1b[31m{repr(e)}\x1b[39m")
            ss["loaderror"] = True
            st.rerun()
    else:
        setup = await setup_chain.ainvoke({"link":link})
    sta.write(f"[{str(datetime.datetime.now().time())[0:8]}] 준비 완료")
    sta.update(label=f"준비 완료", state="complete")
    await talk(setup)
    save_setup = st.chat_input("리뷰 요약본을 파일에 저장하시려면 Y를 입력하세요.")
    if save_setup == 'Y' or save_setup == 'y':
        try:
            with open("setup.dat", "wb") as f:
                f.write(bytes(setup['summary'], encoding="utf-8"))
                f.write(separator)
                f.write(setup['vectorstore'].serialize_to_bytes())
        except:
            print("리뷰 요약본 저장에 실패했습니다.")"""