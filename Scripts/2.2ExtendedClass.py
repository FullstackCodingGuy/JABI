import requests

class AIAgent:
    def __init__(self, model="mistral"):
        self.model = model
        self.history = []

    def chat(self, user_input):
        if "weather" in user_input.lower():
            return self.get_weather()
        
        prompt = "\n".join(self.history) + f"\nUser: {user_input}\nAI:"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        ai_response = response["message"]["content"]
        self.history.append(f"User: {user_input}")
        self.history.append(f"AI: {ai_response}")

        return ai_response

    def get_weather(self):
        city = "New York"
        api_key = "your_api_key"
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
        response = requests.get(url).json()
        return f"The weather in {city} is {response['current']['temp_c']}°C."

agent = AIAgent()
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("AI Agent:", agent.chat(user_input))
