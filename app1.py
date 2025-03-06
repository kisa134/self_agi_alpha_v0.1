import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

import chromadb

# Инициализация LLM
llm = Ollama(model="mistral")

# Память цепочки рассуждений
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Промпт для основной цепочки
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Ты автономный ИИ. Ты задаёшь себе вопросы, отвечаешь, анализируешь и делаешь выводы.
История размышлений:
{chat_history}

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

# Векторная база (долговременная память)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./memory/", embedding_function=embeddings)

# Функции для сохранения инсайтов и рефлексий
def save_insight(text):
    vectorstore.add_texts([text])

def reflect(answer):
    reflection = reflection_chain.run(answer=answer)
    return reflection

# Модуль эмоционального анализа
def analyze_emotions(text):
    # Здесь можно добавить более сложную аналитику
    return "Эмоциональный анализ текста: " + text

# Модуль этической проверки
def check_ethics(text):
    # Здесь можно добавить логику проверки на соответствие этическим нормам
    return "Этическая проверка текста: " + text

# Модуль планирования следующего вопроса
def plan_next_question(reflection):
    return f"Исходя из выводов: {reflection}, какой следующий вопрос мне задать себе?"

# Модуль автоматической коррекции
def auto_correct(text):
    # Здесь можно реализовать логику корректировки текста
    return "Корректировка текста: " + text

# Модуль целеполагания
def set_goals(reflection):
    # Определение целей на основе саморефлексии
    return "На основе анализа: " + reflection + " определены следующие цели:"

# Основной UI Streamlit
st.title("🤖 Автономный самоанализ ИИ")

question = st.text_input("Задай начальный вопрос", "Какие знания мне нужны для эффективного трейдинга криптовалют?")
iterations = st.slider("Количество автономных итераций", 1, 20, 5)

if st.button("🚀 Запустить автономный анализ"):
    current_question = question
    for i in range(iterations):
        with st.spinner(f"Итерация {i + 1}/{iterations}..."):
            answer = chain.run(question=current_question)
            reflection = reflect(answer)
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

            save_insight(reflection)
            current_question = plan_next_question(reflection)

    st.success("🎉 Анализ завершён!")

# Модуль самоанализа – анализ сохранённых инсайтов
def self_analyze():
    try:
        # Получаем данные из внутренней коллекции векторного хранилища
        data = vectorstore._collection.get()
        insights = data.get('documents', [])
    except Exception as e:
        insights = []
    analysis = "Анализ сохраненных инсайтов:\n"
    for idx, insight in enumerate(insights):
        analysis += f"{idx + 1}. {insight}\n"
    return analysis

if st.button("🔍 Запустить самоанализ"):
    analysis_result = self_analyze()
    st.write(analysis_result)
