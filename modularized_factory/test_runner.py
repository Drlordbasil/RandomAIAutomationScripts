import subprocess
import json

def run_tests(script_path):
    """Run tests on the script."""
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            return json.dumps({"result": "All tests passed."})
        else:
            return json.dumps({"error": f"Tests failed:\n{result.stderr}"})
    except Exception as e:
        return json.dumps({"error": f"Error in run_tests: {str(e)}"})
