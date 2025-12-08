from .strategies import MentalHealthChat, HealthAssistantChat

class ChatFactory:
    @staticmethod
    def get_chat_instance(chat_type):
        if chat_type == "mental":
            return MentalHealthChat()
        elif chat_type == "health":
            return HealthAssistantChat()
        else:
            return HealthAssistantChat()  # default
