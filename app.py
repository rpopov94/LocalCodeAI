"""Main API with chunking support."""
import gradio as gr

from bot import RAGBot


bot = RAGBot()

def chat_interface(query: str, history: list[list[str]]):
    bot_response = bot.generate_response(query)
    history.append((query, bot_response))
    return history, ""

with gr.Blocks() as app:
    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(label="Ваш вопрос")
    clear = gr.Button("Очистить историю")

    msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

app.launch(server_port=7860)
