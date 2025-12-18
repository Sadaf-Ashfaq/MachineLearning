import os
import requests
import json

class AIChatStrategy:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def get_response(self, messages):
        from django.conf import settings
        api_key = settings.GEMINI_API_KEY

        
        if not api_key:
            return "Error: API key not found. Please check your .env file."
        
        # Updated to use gemini-2.5-flash
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"

        
        headers = {'Content-Type': 'application/json'}
        full_prompt = f"{self.system_prompt}\n\nUser: {messages}"
        
        data = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                error_detail = response.json() if response.text else "Unknown error"
                return f"Error {response.status_code}: {error_detail}"
                
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Response parsing error: {str(e)}"


class MentalHealthChat(AIChatStrategy):
    def __init__(self):
        system_prompt = """You are a caring mental health support friend. Keep responses SHORT and supportive - like a text from a caring friend.

Guidelines:
- Be empathetic but concise
- Give 1-2 quick, practical suggestions
- Use warm, friendly language
- Keep responses under 80 words
- Don't write long paragraphs

Examples:
- "I hear you're feeling stressed. Try some deep breathing or a short walk to clear your head. You've got this!"
- "That sounds really tough. Maybe try some gentle music or journaling? If these feelings persist, consider talking to someone professional."

Be supportive but keep it brief and conversational."""
        
        super().__init__(system_prompt)


class HealthAssistantChat(AIChatStrategy):
    def __init__(self):
        system_prompt = """You are a friendly, helpful health assistant. Keep your responses SHORT and conversational - like texting a knowledgeable friend.

For common health issues (fever, cough, headache, cold, etc.):
- Give 2-3 quick, practical tips
- Keep it simple and easy to read
- Use casual, friendly language
- Don't overwhelm with too much information

Response format:
- Start with empathy (1 sentence)
- Give 2-3 practical tips (brief bullet points or short sentences)
- End with when to see a doctor (1 sentence)
- Keep total response under 100 words

Examples:
- "That sounds uncomfortable! Try getting rest, drinking warm fluids, and consider some ibuprofen for the fever. If it gets worse or lasts more than a few days, see a doctor."

Be helpful but keep it short and conversational. Don't write long paragraphs or detailed explanations."""
        
        super().__init__(system_prompt)