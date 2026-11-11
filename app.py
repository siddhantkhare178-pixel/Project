try:
    import spaces  

    @spaces.GPU
    def _gpu_placeholder():
        return "ok"
except ImportError:
    pass  

import gradio as gr
from agent import ask

def chat(message, history, request: gr.Request):
    
    return ask(message, session_id=request.session_hash or "web")

demo = gr.ChatInterface(chat, title="Agentic RAG Assistant", api_name="ask")

if __name__ == "__main__":
    demo.launch()