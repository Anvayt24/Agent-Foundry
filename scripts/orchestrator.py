"""Centralized orchestrator: Planner -> Worker -> Verifier run sequentially."""
import json
import logging

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from agents.planner import create_planner
from agents.worker import create_worker
from agents.verifier import create_verifier
from core.central import run_agent

logger = logging.getLogger(__name__)


def _parse_subtasks(plan_raw: str) -> list[str]:
    """Parse the planner's output into a list of subtasks.

    Falls back to line-splitting if the planner did not return valid JSON.
    """
    try:
        subtasks = json.loads(plan_raw).get("subtasks", [])
        if subtasks:
            return subtasks
    except (json.JSONDecodeError, AttributeError):
        pass
    return [s.strip("-• ").strip() for s in plan_raw.splitlines() if s.strip()]


def orchestrate(user_request: str) -> str:
    planner = create_planner()
    worker = create_worker()
    verifier = create_verifier()

    plan_raw = run_agent(planner, user_request)
    subtasks = _parse_subtasks(plan_raw)

    worker_outputs = []
    for i, task in enumerate(subtasks, 1):
        result = run_agent(worker, task)
        worker_outputs.append(f"[Subtask {i}] {task}\n{result}")

    bundle = "\n\n".join(worker_outputs)
    return run_agent(verifier, bundle)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    while True:
        try:
            query = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit", "quit"):
            break
        try:
            print("Final Answer:", orchestrate(query))
        except Exception:
            logger.exception("Error during orchestration")


if __name__ == "__main__":
    main()
