
import sqlite3
import subprocess
from flask import Flask, request

app = Flask(__name__)

# Vulnerability 1: SQL Injection
@app.route("/user")
def get_user():
    username = request.args.get("username")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    # Vulnerable: User input is directly added to SQL
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

    result = cursor.fetchall()
    conn.close()

    return str(result)


# Vulnerability 2: Command Injection
@app.route("/ping")
def ping():
    host = request.args.get("host")

    # Vulnerable: User input is passed to a shell command
    result = subprocess.run(
        f"ping -c 1 {host}",
        shell=True,
        capture_output=True,
        text=True
    )

    return result.stdout


# Vulnerability 3: Cross-Site Scripting (XSS)
@app.route("/greet")
def greet():
    name = request.args.get("name", "")

    # Vulnerable: Untrusted input is returned as raw HTML
    return f"<h1>Hello, {name}!</h1>"


if __name__ == "__main__":
    app.run(debug=False)
