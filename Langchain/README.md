Run the project
```
uvicorn agent_chatbot:app --host 0.0.0.0 --port 8000          
```

Sample Request Payloads

```
{
    "message3": "Who won the FIFA World Cup 2022?",
    "message": "run python print(15+3)",
    "message1": "search latest AI trends",
    "chat_history": [
        {
            "user": "Who won the FIFA World Cup 2022?",
            "bot": "Argentina won the FIFA World Cup 2022."
        }
    ]
}
```