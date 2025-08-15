import json
from planner import create_planner
from worker import create_worker
from verifier import create_verifier
from central import run_agent
from messaging import Messenger, Message

def orchestrate(user_request: str) -> str:
    messenger = Messenger()  #messenger call

    planner = create_planner()           #instantiating agents
    worker = create_worker()
    verifier = create_verifier()

    plan_raw = messenger.send_direct(
        lambda payload: run_agent(planner, payload),
        Message(sender="user", recipient="planner", payload=user_request)
    )
    try:
        subtasks = json.loads(plan_raw).get("subtasks", [])
    except Exception:
        subtasks = [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]

    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        result = messenger.send_direct(           # worker output
            lambda payload: run_agent(worker, payload),
            Message(sender="planner", recipient="worker", payload=task)
        )
        worker_outputs.append(f"[Subtask {i}] {task}\n{result}")   

    bundle = "\n\n".join(worker_outputs)
    final_answer = messenger.send_direct(      # verifier output
        lambda payload: run_agent(verifier, payload),
        Message(sender="worker", recipient="verifier", payload=bundle)
    )

    return final_answer

if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ("exit", "quit"):
            break
        print("Final Answer:", orchestrate(query))

