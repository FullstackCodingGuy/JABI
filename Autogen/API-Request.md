# Run the assistant application
```
uvicorn main:app --reload
```

## Use the below request to send the Prompt to assistant
```
curl --location 'http://localhost:8000/chat' \
--header 'Content-Type: application/json' \
--data '{
    "message": "I want to know about the latest advancements in AI technology.",
}'
```