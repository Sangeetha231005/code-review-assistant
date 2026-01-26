import subprocess
import sqlite3
from typing import Optional


def safe_echo(user_input: str) -> None:
    """
    Safely echo user input without invoking a shell.
    """
    # Use argument list instead of shell=True
    subprocess.run(
        ["echo", user_input],
        check=True
    )


def get_user_by_name(username: str) -> Optional[tuple]:
    """
    Safely query a database using parameterized SQL.
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

    # ✅ SAFE: parameterized query
    cursor.execute(
        "SELECT * FROM users WHERE name = ?",
        (username,)
    )

    result = cursor.fetchone()
    conn.close()
    return result


def list_directory(path: str) -> None:
    """
    Safely list directory contents without shell execution.
    """
    subprocess.run(
        ["ls", path],
        check=True
    )


def main() -> None:
    user_input = "test_input"

    safe_echo(user_input)

    user = get_user_by_name("admin")
    print("Query result:", user)

    list_directory(".")


if __name__ == "__main__":
    main()
