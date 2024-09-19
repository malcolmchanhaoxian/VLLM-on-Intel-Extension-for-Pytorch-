import argparse
import time
import gradio as gr
from openai import OpenAI

parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--url',
                    type=str,
                    default='http://localhost:8000/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--max_tokens',
                    type=int,
                    default=128,
                    help='Maximum number of tokens for text generation')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8000)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
# Since we are not accessing any direct models from OpenAI service, api key is left as empty
client = OpenAI(
    api_key="EMPTY",
    base_url= args.url,
)

def user(user_prompt, history):
    return "", history + [[user_prompt, None]]

def generate_response(history):
    user_message = history[-1][0]
    response = [{"role": "user", "content": user_message}]
    completion = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=response,  # Chat history
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=args.max_tokens
    )

    bot_response = ""
    tokens_num = 0
    st = time.time()

    for chunk in completion:
      if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason is not None:
        next_chunk = next(completion, None)
        if next_chunk:
          tokens_num = next_chunk.usage.completion_tokens
          et = time.time() - st
          yield history, f'{round(tokens_num/et,2)}'
        else:
          return None # Handle the case where there's no next chunk
      else:
        et = time.time() - st
        chunk_message = chunk.choices[0].delta.content or ""
        bot_response += chunk_message
        history[-1][1] = bot_response
        yield history, f'{round(tokens_num/et,2)}'

with gr.Blocks(theme=gr.Theme.from_hub('HaleyCH/HaleyCH_Theme')) as demo:

    gr.Markdown("""
    <h1><center>Intel Chatbot with vLLM Model Serving
    <center><img src="https://upload.wikimedia.org/wikipedia/commons/6/64/Intel-logo-2022.png" width=200px>
    <h2>Inferenced on Dell Poweredge powered by Intel 5th Gen Xeon with AMX Acceleration</h2></center>
    """)

    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation")

    with gr.Row():
        with gr.Column(scale=5): prompt_input = gr.Textbox(label="Enter your message here", placeholder="Type your message...")
        with gr.Column(scale=1): token_output = gr.Textbox(label="Tokens/sec", placeholder="0.00")

    prompt_input.submit(user, [prompt_input, chatbot], [prompt_input, chatbot], queue=False).then(
        generate_response, inputs=chatbot, outputs=[chatbot,token_output], queue=True)

    gr.Markdown("""
    <center><br><h3>An Intel Collaboration by Beny Ibrani // Malcolm Chan</h3>
    """)
    
demo.queue().launch(debug=True, share=True, server_name=args.host, server_port=args.port)
