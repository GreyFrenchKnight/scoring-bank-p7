CALL .env/Scripts/activate.bat

@echo off
uvicorn app:app

cmd \k