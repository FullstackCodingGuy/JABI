# Example 1
Run the `agent_chatbot.py` project
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

# Example 2
Run the `agent_chatbot_rag.ph` project
```
uvicorn agent_chatbot_rag:app --host 0.0.0.0 --port 8000
```

### General Chat
```
{"message": "Who won the FIFA World Cup 2022?", "chat_history": []}
```
Expected Output: "The FIFA World Cup 2022 was won by Argentina."

### Multi-turn Conversation
```
{
    "message": "Tell me a joke",
    "chat_history": [{"user": "Who won the FIFA World Cup 2022?", "bot": "Argentina won the FIFA World Cup 2022."}]
}
```
Expected Output: A joke + chat history retained.

### Retrieve from Vector DB
```
{"message": "Who is the CEO of Tesla?", "chat_history": []}
```
Expected Output: "Elon Musk is the CEO of Tesla and SpaceX."


