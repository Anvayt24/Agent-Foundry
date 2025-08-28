import json
import requests  
from worker import create_worker
from verifier import create_verifier
from central import run_agent


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
            {"role": "user", "content": user_request}
        ]
    }
).json()

        subtasks = agc_resp.get("steps", [])
        if not subtasks:
            plan_raw = agc_resp.get("output", user_request)
            subtasks = [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]
    except Exception as e:
        print("Agc planner failed,back to simple split:", e)
        subtasks = [user_request]
    
    #tool testing
    print("\n tool choosen ")
    for i, step in enumerate(subtasks, 1):
        tool = step.get("tool", "unknown") if isinstance(step, dict) else "not_structured" #capturing tool call
        content = step.get("input", step) if isinstance(step, dict) else step
        print(f"Step {i}: Tool = {tool}, Input = {content}")    

    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        task_input = task.get("input") if isinstance(task, dict) else task
        result = run_agent(worker, task_input)
        worker_outputs.append(f"[Subtask {i}] {task_input}\n{result}")

    #verifier
    bundle = "\n\n".join(worker_outputs)
    final_answer = run_agent(verifier, bundle)

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





