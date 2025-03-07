import streamlit as st
import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import weaviate

# Инициализация LLM
llm = Ollama(model="mistral")

# Память цепочки рассуждений
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",  # ВАЖНО!
    return_messages=True
)

# Embeddings для Weaviate
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Подключение к Weaviate v4
client = weaviate.connect_to_local()
collection = client.collections.get("Insights")

# Промпт для основной цепочки с RAG
prompt = PromptTemplate(
    input_variables=["chat_history", "question", "insights"],
    template="""Ты автономный ИИ. Ты задаёшь себе вопросы, отвечаешь, анализируешь и делаешь выводы.

История размышлений:
{chat_history}

Релевантные инсайты из долговременной памяти:
{insights}

Вопрос: {question}
Ответ и подробное рассуждение:"""
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Промпт для саморефлексии
reflection_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""Проанализируй свой ответ:
Ответ: {answer}

Что можно улучшить? Сформулируй критику и улучшение:"""
)

reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt)



def save_insight(text):
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace("+00:00", "Z")
    collection.data.insert({
        "content": text,
        "timestamp": timestamp
    })



# Извлечение релевантных инсайтов (RAG)
def retrieve_insights(query, top_k=3):
    query_vector = embeddings.embed_query(query)
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_properties=["content"]
    )
    insights = [obj.properties['content'] for obj in response.objects]
    return insights if insights else ["Инсайты не найдены"]

# Дополнительные модули
def analyze_emotions(text):
    return "Эмоциональный анализ текста: " + text

def check_ethics(text):
    return "Этическая проверка текста: " + text

def plan_next_question(reflection):
    return f"Исходя из выводов: {reflection}, какой следующий вопрос мне задать себе?"

def auto_correct(text):
    return "Корректировка текста: " + text

def set_goals(reflection):
    return "На основе анализа: " + reflection + " определены следующие цели:"

# Основной UI Streamlit
st.title("🤖 Автономный самоанализ ИИ")

question = st.text_input("Задай начальный вопрос", "Какие знания мне нужны для эффективного трейдинга криптовалют?")
iterations = st.slider("Количество автономных итераций", 1, 20, 5)

if st.button("🚀 Запустить автономный анализ"):
    current_question = question
    for i in range(iterations):
        with st.spinner(f"Итерация {i + 1}/{iterations}..."):
            insights = retrieve_insights(current_question)
            insights_text = "\n".join(insights for insights in insights) if insights else "Нет релевантных инсайтов."

            answer = chain.run(question=current_question, insights="\n".join(insights))
            reflection = reflection_chain.run(answer=answer)
            emotions = analyze_emotions(answer)
            ethics = check_ethics(answer)
            correction = auto_correct(answer)
            goals = set_goals(reflection)

            st.subheader(f"Итерация {i + 1}")
            st.write(f"**🔸 Вопрос:** {current_question}")
            st.write(f"**💬 Ответ модели:** {answer}")
            st.write(f"**🧠 Саморефлексия:** {reflection}")
            st.write(f"**🎯 Эмоциональный анализ:** {emotions}")
            st.write(f"**⚖️ Этическая оценка:** {ethics}")
            st.write(f"**🔄 Автоматическая коррекция:** {correction}")
            st.write(f"**🎯 Цели:** {goals}")

            # Сохраняем инсайты и ответы
            save_insight(reflection)
            save_insight(answer)

            current_question = plan_next_question(reflection)

    st.success("🎉 Анализ завершён!")

# Модуль самоанализа – анализ сохранённых инсайтов
def self_analyze():
    insights = collection.query.fetch_objects(limit=100)
    analysis = "Анализ сохраненных инсайтов:\n"
    for idx, insight in enumerate(insights.objects):
        analysis += f"{idx + 1}. {insight.properties['content']}\n"
    return analysis

if st.button("🔍 Запустить самоанализ"):
    analysis_result = self_analyze()
    st.write(analysis_result)
