import time
from messaging import MessageBus
from agents.planner import create_planner_a2a,PlannerA2A
from agents.worker import create_worker_a2a, WorkerA2A
from agents.verifier import create_verifier_a2a, VerifierA2A

class A2ANetwork:
    def __init__(self):
        self.message_bus = MessageBus()
        self.planner: PlannerA2A = create_planner_a2a(self.message_bus)
        self.worker: WorkerA2A = create_worker_a2a(self.message_bus)
        self.verifier: VerifierA2A = create_verifier_a2a(self.message_bus)

    def run(self, user_input: str) -> str:
        # Planner gets user input and sends subtasks to Worker
        self.planner.process_user_request(user_input)

        # Simple cooperative pump loop
        idle_rounds = 0
        max_idle_rounds = 10
        end_time = time.time() + 10.0  # hard cap to avoid runaway loops
        while time.time() < end_time and idle_rounds < max_idle_rounds:
            progressed = False
            progressed |= self.worker.process_once(timeout=0.1)
            progressed |= self.verifier.process_once(timeout=0.1)
            if progressed:
                idle_rounds = 0
            else:
                idle_rounds += 1

        # Planner collects final results
        final = self.planner.collect_results(max_wait_seconds=1.0)
        return final

if __name__ == "__main__":
    a2a_network = A2ANetwork()
    try:
        while True:
            user_input = input("User (A2A)> ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            result = a2a_network.run(user_input)
            print("\nFinal Answer:\n", result or "<no result>", "\n")
    except KeyboardInterrupt:
        pass
