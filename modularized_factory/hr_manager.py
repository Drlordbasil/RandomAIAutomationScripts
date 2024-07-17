import json

def manage_hr(action, agent_name):
    """Manage HR tasks such as hiring or firing agents."""
    try:
        if action == 'hire':
            return json.dumps({"result": f"HR tasks completed: {agent_name} has been hired."})
        else:
            return json.dumps({"result": f"HR tasks completed: {agent_name} has been fired."})
    except Exception as e:
        return json.dumps({"error": f"Error in manage_hr: {str(e)}"})
