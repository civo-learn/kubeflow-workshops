FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

COPY app.py .

RUN apt-get update && \
    apt-get install -y wget && \
    pip install --no-cache-dir -r requirements.txt

RUN wget https://huggingface.co/TheBloke/Llama-2-7B-chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_1.bin


# EXPOSE 8080

CMD ["python", "./app.py"]
