import os
import gradio as gr
import requests
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# For local development, uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Using Qwen 2.5 72B - one of the best open-source models
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class MistralEmbeddings(Embeddings):
    """Mistral embeddings to match the original index"""
    
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = "mistral-embed"
        self.url = "https://api.mistral.ai/v1/embeddings"
    
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self._get_embeddings([text])[0]


class ResearchParrot:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self._vectorstore = None
        self._embeddings = None

    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = MistralEmbeddings()
        return self._embeddings

    def vectorstore(self):
        if self._vectorstore is None:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("parrot")
            self._vectorstore = PineconeVectorStore(embedding=self.embeddings(), index=index)
        return self._vectorstore

    def query(self, question: str):
        if not question.strip():
            return "Please enter a question."
        
        store = self.vectorstore()
        docs = store.similarity_search(question, k=5)
        
        if not docs:
            return "No relevant documents found in the knowledge base."
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a research assistant. Answer the question ONLY based on the provided context. 

IMPORTANT RULES:
- Only use information from the context below and Make a Step by Step approach.
- If the context doesn't contain enough information to answer, say "I don't have enough information in the documents to answer this question."
- Always make it more technical in depth as much as you can because your readers are security researchers not normal people.
- Always highlight the attack technique, payload, math formula properly if available.

Context:
{context}

Question: {question}

Answer:"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            model=self.model_id,
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content


# Initialize the app lazily to show better errors
app = None


def get_app():
    global app
    if app is None:
        app = ResearchParrot()
    return app


def chat(message, history):
    """Chat function for the Gradio interface"""
    try:
        response = get_app().query(message)
        return response
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error: {type(e).__name__}: {str(e)}\n\nDetails:\n{error_details}"


# Build Gradio Interface for Hugging Face Spaces
with gr.Blocks(theme=gr.themes.Soft(), title="Research Parrot") as demo:
    gr.Markdown(
        """
        # Research Parrot
        ### AI-Powered Research Paper Assistant
        
        Ask questions about security research papers.
        Perfect for security researchers who need in-depth technical analysis.
        """
    )
    
    chatbot = gr.Chatbot(
        label="Research Assistant",
        height=500,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
            {"left": "\\[", "right": "\\]", "display": True},
            {"left": "\\(", "right": "\\)", "display": False},
        ]
    )
    
    msg = gr.Textbox(
        label="Your Question",
        placeholder="Ask about security research...",
        lines=2
    )
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    gr.Examples(
        examples=[
            "Tell me about jailbreaking?",
            "What is prompt injection?",
            "Explain RAG architecture"
        ],
        inputs=msg
    )
    
    def respond(message, chat_history):
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: None, None, chatbot, queue=False)

# Launch configuration for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )