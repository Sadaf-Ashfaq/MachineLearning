from django.shortcuts import render, redirect
from .chat_factory import ChatFactory
from .models import ChatMessage
from django.contrib.auth.decorators import login_required

def chat_router(request):
    if request.method == 'POST':
        chat_type = request.POST.get('chat_type')
        return redirect('chat', chat_type=chat_type)
    return render(request, 'chatbot/chat_select.html')

@login_required
def chat_view(request, chat_type):
    user_id = str(request.user.id)
    history = ChatMessage.objects.filter(user_id=user_id, chat_type=chat_type).order_by("timestamp")
    messages = [{"role": msg.role, "text": msg.text} for msg in history]

    if request.method == 'POST':
        user_msg = request.POST.get("message")
        ChatMessage.objects.create(user_id=user_id, chat_type=chat_type, role="user", text=user_msg)

        bot = ChatFactory.get_chat_instance(chat_type)
        history_payload = [{"role": "user", "parts": [msg["text"]]} for msg in messages[-5:]]
        history_payload.append({"role": "user", "parts": [user_msg]})
        bot_reply = bot.get_response(history_payload)

        ChatMessage.objects.create(user_id=user_id, chat_type=chat_type, role="bot", text=bot_reply)
        return redirect("chat_view", chat_type=chat_type)

    return render(request, "chatbot/chat.html", {
        "chat_type": chat_type,
        "history": history
    })

@login_required
def clear_chat_view(request, chat_type):
    if request.method == "POST":
        ChatMessage.objects.filter(user_id=str(request.user.id), chat_type=chat_type).delete()
        return redirect('chat_view', chat_type=chat_type)
