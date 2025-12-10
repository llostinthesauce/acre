# ACRE LLM Switchboard

## HOW TO USE:

1. Clone the repository from GitHub to your device.

2. Ensure that Python is installed, version 3.11+ is required to run the application.

3. Open a terminal and proceed to the location where you cloned the repository.

3.9. Optionally, but recommended, run in a virtual environment
"python3 -m venv .venv && source .venv/bin/activate"

4. To ensure that you have installed the required dependencies, run the following command:
"pip install -r requirements.txt"
This command will install all of the required dependencies to run the app, which are listed in the requirements.txt file.

6. You will now be able to run the application in offline mode. Depending on your preferences, you may disconnect from your network or stay connected. In order to
download models to run, proceed to step 7 before disconnecting your device from the network.

7. In the terminal, enter the following command:
"python app.py"
If that does not work, try the following instead:
"py app.py" or "python3 app.py"
The application will now run on your device.

8. To download models, visit "https://huggingface.co/", or click the link located within the settings menu in the application.
reccomended for testing is this: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

10. You may now add/load models. For easy access to models, place them in the models folder located within the application directory.
