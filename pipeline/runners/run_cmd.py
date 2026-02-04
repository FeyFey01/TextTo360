import subprocess
import json

def run_cmd_live_json(cmd):
    """
    Runs a command with live printing in terminal.
    Expects the last line of stdout to be JSON.
    Returns the parsed JSON.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    last_line = ""
    for line in process.stdout:
        print(line, end="")   # live terminal output
        last_line = line      # keep updating last_line

    process.wait()

    # Parse the last line as JSON
    try:
        return json.loads(last_line.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from last line: {last_line}") from e
