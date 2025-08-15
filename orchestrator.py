import json
from planner import create_planner
from worker import create_worker
from verifier import create_verifier

def orchestrate(user_request: str) -> str:  
    planner = create_planner()        #instantiating agents
    worker = create_worker()
    verifier = create_verifier()

    plan_raw = planner.invoke({"input": user_request})["output"]   #planner output
    try:
        subtasks = json.loads(plan_raw).get("subtasks", [])       
    except Exception:
        subtasks = [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]

    if not subtasks:
        return "Planner produced no subtasks."

    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        res = worker.invoke({"input": task})["output"]   #worker output  
        worker_outputs.append(f"[Subtask {i}] {task}\n{res}")

    bundle = "\n\n".join(worker_outputs)
    final_answer = verifier.invoke({"input": bundle})["output"]   #verifier output
    return final_answer

if __name__ == "__main__":
    print("AgentFoundry Orchestrator (Planner -> Worker -> Verifier)\n")
    try:
        while True:
            q = input("User: ")
            if q.lower() in ["exit", "quit"]: break
            print("\n[Running pipeline...]\n")
            out = orchestrate(q)
            print("\n=== Final Answer ===\n", out, "\n")
    except KeyboardInterrupt:
        pass


