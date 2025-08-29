import json
import requests  
from worker import create_worker
from verifier import create_verifier
from central import run_agent

LOG_FILE = "tool_output.json"


def save_log_entry(query, subtasks, final_answer):
    """Append planner decisions + result to JSON log file."""
    entry = {
        "query": query,
        "subtasks": subtasks,
        "final_answer": final_answer
    }

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def orchestrate(user_request: str) -> str:
    worker = create_worker()
    verifier = create_verifier()
    
    #AgC as planner 
    try:
        agc_resp = requests.post(
    "http://localhost:6644/v1/responses",
    json={
        "model": "google@gemini-2.5-flash",
        "input": [
      {
          "role": "system",
          "content": "You are a planner agent. Break down the user query and decide which tool to call. For each step, return JSON with fields: {\"tool\": <tool_name>, \"input\": <query for tool>}."
      },
      {
          "role": "user",
          "content": user_request
      }
  ]
}).json()

        subtasks = agc_resp.get("output", {}).get("steps", [])
        if not subtasks:
            plan_raw = agc_resp.get("output", user_request)
            subtasks = [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]
    except Exception as e:
        print("Agc planner failed,back to simple split:", e)
        subtasks = [user_request]
    
    #tool testing
    print("\n[Planner Decisions]")
    normalized_subtasks = []
    for i, step in enumerate(subtasks, 1):
     if isinstance(step, dict):
        tool = step.get("tool", "unknown")
        content = step.get("input", "")
    else:  # step is a plain string 
        tool = "not_structured"
        content = str(step)
    print(f"Step {i}: Tool = {tool}, Input = {content}")
    normalized_subtasks.append({"tool": tool, "input": content})

    subtasks = normalized_subtasks

    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        task_input = task.get("input") if isinstance(task, dict) else task
        result = run_agent(worker, task_input)
        worker_outputs.append(f"[Subtask {i}] {task_input}\n{result}")

    #verifier
    bundle = "\n\n".join(worker_outputs)
    final_answer = run_agent(verifier, bundle)

    save_log_entry(user_request, subtasks, final_answer)

    return final_answer


if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ("exit", "quit"):
            break
        try:
            print("Final Answer:", orchestrate(query))
        except Exception as e:
            print("Error in orchestration:", e)





