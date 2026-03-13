from flask import Flask, request
from pyngrok import ngrok

ngrok.set_auth_token("3AtbqRbtKoMcQHExzTMLaA7dTCr_Sb8BJEALNzYfmQiDnpwJ")

app = Flask(__name__)
losses = []

@app.route("/log", methods=["POST"])
def log():
    losses.append(request.json["loss"])
    print(losses)
    
    return "ok"

public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

app.run(port=5000)