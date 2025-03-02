@echo off
echo Запускаю Ollama сервер...
start ollama serve

timeout /t 5

echo Запускаю Streamlit приложение...
streamlit run app.py
pause
