from flask import Flask, request
from ctransformers import AutoModelForCausalLM
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

model_id = "llama-2-7b-chat.ggmlv3.q4_1.bin"

config = {
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "stream": True,
}
llm = AutoModelForCausalLM.from_pretrained(
    model_id, model_type="llama", lib="avx2", gpu_layers=110, **config
)


@app.route("/v1/models/serving:predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data["prompt"]
    if data["stream"]:
        tokens = llm.tokenize(prompt)
        response = ""
        for token in llm.generate(tokens):
            response += llm.detokenize(token)
    else:
        response = llm(prompt, stream=False)
    return response


if __name__ == "__main__":
    http_server = WSGIServer(("", 8080), app)
    http_server.serve_forever()
