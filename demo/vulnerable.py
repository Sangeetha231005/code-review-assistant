import os

def run_command(user_input):
    # VULNERABLE: Command Injections
    os.system("echo " + user_input)

if __name__ == "__main__":
    user = input("Enter command: ")
    run_command(user)
