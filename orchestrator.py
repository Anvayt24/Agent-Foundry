import json
from planner import create_planner
from worker import create_worker
from verifier import create_verifier
from central import run_agent

def orchestrate(user_request: str) -> str:
    # Create agents
    planner = create_planner()
    worker = create_worker()
    verifier = create_verifier()

    # Planner
    plan_raw = run_agent(planner, user_request)

    # Parsing subtasks
    try:
        subtasks = json.loads(plan_raw).get("subtasks", [])
    except Exception:
        subtasks = [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]

    #Workers
    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        result = run_agent(worker, task)
        worker_outputs.append(f"[Subtask {i}] {task}\n{result}")

    #Verifier
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



