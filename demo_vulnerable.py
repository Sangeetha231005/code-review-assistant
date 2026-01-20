import os
import sqlite3

def vulnerable_function(user_input):
    # ❌ Command Injection
    os.system("echo " + user_input)

    # ❌ SQL Injection
    conn = sqlite3.connect(":memory:")
    conn.execute("SELECT * FROM users WHERE id = " + user_input)

def main():
    user_input = input("Enter ID: ")
    vulnerable_function(user_input)

if __name__ == "__main__":
    main()
