CALL venv/Scripts/activate.bat

@echo off
uvicorn app:app --reload

cmd \k