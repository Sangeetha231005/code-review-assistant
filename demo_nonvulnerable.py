"""
Non-vulnerable Python module demonstrating secure coding standards.
- Parameterized queries to prevent SQL Injection
- Argument lists to prevent Command Injection
- Proper typing and input validation
"""
import subprocess
import sqlite3
from typing import Optional


def safe_echo(user_input: str) -> None:
    """
    Safely echo user input without invoking a shell.
    Uses argument list instead of shell=True to eliminate command injection.
    """
    # ✅ SAFE: No shell=True, input cannot execute sub-commands
    subprocess.run(
        ["echo", user_input],
        check=True
    )


def get_user_by_name(username: str) -> Optional[tuple]:
    """
    Safely query database using parameterized SQL.
    Prevents SQL injection by treating input strictly as a parameter value.
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create sample table
    cursor.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"
    )
    cursor.execute(
        "INSERT INTO users (name) VALUES (?)",
        ("admin",)
    )

    # ✅ SAFE: Parameterized query using '?' placeholder
    cursor.execute(
        "SELECT * FROM users WHERE name = ?",
        (username,)
    )

    result = cursor.fetchone()
    conn.close()
    return result


def list_directory(path: str) -> None:
    """
    Safely list directory contents using safe subprocess parameters.
    """
    # ✅ SAFE: Direct binary call without shell execution
    subprocess.run(
        ["ls", path],
        check=True
    )


def main() -> None:
    user_input = "test_input"

    # 1. Run safe command execution
    safe_echo(user_input)

    # 2. Run safe parameterized database query
    user = get_user_by_name("admin")
    print("Query result:", user)

    # 3. Run safe directory listing
    list_directory(".")


if __name__ == "__main__":
    main()
