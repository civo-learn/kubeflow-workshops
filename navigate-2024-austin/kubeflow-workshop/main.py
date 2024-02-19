import os
import json
import fastapi
import uvicorn
import concurrent.futures
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel, Field
from typing import List, Generator
import logging
import asyncio
from datetime import datetime


config = {
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
}
llm = AutoModelForCausalLM.from_pretrained("llama-2-7b-chat.ggmlv3.q4_1.bin",
                                           model_type="llama",
                                           lib="avx2",
                                           gpu_layers=110, 
                                           threads=8,
                                           context_length = 4096,
                                           **config)
app = fastapi.FastAPI(title="Llama 2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    html_content = """
    <html>
        <head>
        </head>
        <body>
            <h2>Run Llama 2</h2>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

class ChatCompletionRequestV0(BaseModel):
    prompt: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512

@app.post("/v1/completions")
async def completion(request: ChatCompletionRequestV0, response_mode=None):
    response = llm(request.prompt)
    return response

def generate_response(chat_chunks, llm):
    response = ""
    for chat_chunk in chat_chunks:
        response += llm.detokenize(chat_chunk)
    response = {
        "created": str(datetime.now()),
        "model": "llama-2",     
        'choices': [
            {
                'message': {
                    'role': 'system',
                    'content': response
                },
                'finish_reason': 'stop'
            }
        ]
    }
    return response

@app.post("/v1/chat/completions")
def chat(request: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in request.messages])
    tokens = llm.tokenize(combined_messages)
    
    try:
        chat_chunks = llm.generate(tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return generate_response(chat_chunks, llm)

async def stream_response(tokens, llm):
    try:
        iterator: Generator = llm.generate(tokens)
        for chat_chunk in iterator:
            response = {
                'choices': [
                    {
                        'message': {
                            'role': 'system',
                            'content': llm.detokenize(chat_chunk)
                        },
                        'finish_reason': 'stop' if llm.is_eos_token(chat_chunk) else 'unknown'
                    }
                ]
            }
            yield json.dumps(response)
    except Exception as e:
        print(f"Exception in event publisher: {str(e)}")

@app.post("/v2/chat/completions")
async def chatV2_endpoint(request: Request, body: ChatCompletionRequest):
    combined_messages = ' '.join([message.content for message in body.messages])
    tokens = llm.tokenize(combined_messages)

    return EventSourceResponse(stream_response(tokens, llm))
        
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)