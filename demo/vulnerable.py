"""
vulnerable.py

This file contains intentionally vulnerable code
to test the Code Review Assistant pipeline.
DO NOT USE IN PRODUCTION.
"""

import os
import sqlite3
import subprocess


def command_injection(user_input):
    """
    ❌ VULNERABILITY: Command Injection
    User input is passed directly to a shell command.
    """
    os.system("echo " + user_input)  # CRITICAL: command injection


def sql_injection(username):
    """
    ❌ VULNERABILITY: SQL Injection
    User input is concatenated into a SQL query.
    """
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)  # CRITICAL: SQL injection

    result = cursor.fetchall()
    conn.close()
    return result


def unsafe_subprocess(user_input):
    """
    ❌ VULNERABILITY: Shell Injection via subprocess
    """
    subprocess.Popen(
        "ls " + user_input,
        shell=True
    )


def main():
    user_input = input("Enter input: ")

    command_injection(user_input)
    sql_injection(user_input)
    unsafe_subprocess(user_input)


if __name__ == "__main__":
    main()
