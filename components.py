import os
import requests
from duckduckgo_search import DDGS
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- Calculator Agent ---
class CalculatorAgent:
    def run(self, query):
        if "add" in query.lower():
            nums = [int(s) for s in query.split() if s.isdigit()]
            return str(sum(nums))
        elif "multiply" in query.lower():
            nums = [int(s) for s in query.split() if s.isdigit()]
            result = 1
            for n in nums:
                result *= n
            return str(result)
        return None

# --- Weather Agent ---
class WeatherAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def run(self, query):
        import re
        match = re.search(r'weather in ([\w\s]+)', query.lower())
        if not match:
            return None
        city = match.group(1).strip()
        url = f"https://api.tomorrow.io/v4/weather/realtime?location={city}&apikey={self.api_key}"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            temp = data['data']['values']['temperature']
            cond = data['data']['values']['weatherCode']
            return f"The weather in {city.title()} is {cond} with {temp}°C."
        return "Sorry, couldn't fetch weather."

# --- Search Agent ---
class SearchAgent:
    def run(self, query):
        if "search" in query.lower():
            q = query.lower().replace("search", "").strip()
            with DDGS() as ddgs:
                results = list(ddgs.text(q, max_results=1))
                if results:
                    return results[0]['body']
            return "No results found."
        return None

# --- Chat Agent (LLM) ---

class ChatAgent:
    def __init__(self, hf_token, model="HuggingFaceH4/zephyr-7b-beta"):
        self.client = InferenceClient(token=hf_token)
        self.model = model
        self.memory = ConversationBufferMemory()

    def run(self, query):
        history = self.memory.load_memory_variables({})["history"]
        prompt = f"{history}\nUser: {query}\nAssistant:"
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"Sorry, the model could not generate a response: {e}"
        self.memory.save_context({"input": query}, {"output": result})
        return result

# --- Master Agent ---
class MasterAgent:
    def __init__(self, hf_token, weather_api_key):
        self.calculator = CalculatorAgent()
        self.weather = WeatherAgent(weather_api_key)
        self.search = SearchAgent()
        self.chat = ChatAgent(hf_token)

    def route(self, query):
        for agent in [self.calculator, self.weather, self.search]:
            result = agent.run(query)
            if result:
                return result
        return self.chat.run(query)