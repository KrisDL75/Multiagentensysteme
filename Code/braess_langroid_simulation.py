from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import langroid as lr


@dataclass(frozen=True)
class Edge:
    name: str
    start: str
    end: str
    constant: float = 0.0
    load_factor: float = 0.0
    toll: float = 0.0

    def travel_time(self, load: float) -> float:
        return self.constant + self.load_factor * load

    def decision_cost(self, load: float) -> float:
        return self.travel_time(load) + self.toll


@dataclass
class Decision:
    node: str
    options: list[dict]
    best_edge: str
    llm_choice: Optional[str]
    final_choice: str
    decision_type: str
    llm_raw: str


@dataclass
class AgentResult:
    agent_id: int
    route_nodes: list[str]
    route_edges: list[str]
    final_time: float
    total_toll: float
    decisions: list[Decision]


class DriverAgent:
    def __init__(self, model: str, ollama_url: str) -> None:
        llm_config = lr.language_models.OpenAIGPTConfig(
            chat_model=model,
            api_base=ollama_url,
            api_key="ollama",
            ollama=True,
            temperature=0,
            max_output_tokens=40,
            timeout=180,
            stream=False,
        )
        self.agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                name="braess_driver",
                system_message=(
                    "You are one selfish driver in a traffic simulation. "
                    "Choose exactly one valid edge with the smallest estimated_total_cost. "
                    "The decision cost is travel time plus toll; toll is not travel time. "
                    "Reply only with JSON in the form {\"edge\":\"EDGE_NAME\"}."
                ),
                llm=llm_config,
            )
        )

    def choose(self, node: str, options: list[dict]) -> tuple[Optional[str], str]:
        valid_edges = [option["edge"] for option in options]
        prompt = json.dumps(
            {
                "current_node": node,
                "valid_edges": valid_edges,
                "options": options,
                "instruction": "Select the edge with the smallest estimated_total_cost.",
            }
        )
        try:
            response = self.agent.llm_response(prompt)
            raw = response.content.strip() if response is not None else ""
        except Exception as exc:
            return None, f"LLM_ERROR: {exc}"

        try:
            choice = json.loads(raw).get("edge")
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            choice = json.loads(match.group(0)).get("edge") if match else None

        if choice in valid_edges:
            return choice, raw
        return None, raw


def build_network(extended: bool, toll_ab: float) -> tuple[dict[str, Edge], dict[str, list[str]]]:
    edges = {
        "S_A": Edge("S_A", "Start", "A", load_factor=1 / 100),
        "A_Z": Edge("A_Z", "A", "Ziel", constant=45),
        "S_B": Edge("S_B", "Start", "B", constant=45),
        "B_Z": Edge("B_Z", "B", "Ziel", load_factor=1 / 100),
    }
    if extended:
        edges["A_B"] = Edge("A_B", "A", "B", constant=2, toll=toll_ab)

    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edges.values():
        graph[edge.start].append(edge.name)
    return edges, graph


def shortest_remaining_cost(
    node: str,
    edges: dict[str, Edge],
    graph: dict[str, list[str]],
    loads: dict[str, float],
    vehicle_weight: float,
) -> float:
    if node == "Ziel":
        return 0.0

    best: dict[str, float] = {node: 0.0}
    pending = [node]
    while pending:
        current = min(pending, key=best.__getitem__)
        pending.remove(current)
        for edge_name in graph.get(current, []):
            edge = edges[edge_name]
            candidate = best[current] + edge.decision_cost(loads[edge_name] + vehicle_weight)
            if candidate < best.get(edge.end, float("inf")):
                best[edge.end] = candidate
                pending.append(edge.end)
    return best.get("Ziel", float("inf"))


def make_options(
    node: str,
    edges: dict[str, Edge],
    graph: dict[str, list[str]],
    loads: dict[str, float],
    vehicle_weight: float,
) -> list[dict]:
    options = []
    for edge_name in graph[node]:
        edge = edges[edge_name]
        projected_load = loads[edge_name] + vehicle_weight
        projected_loads = dict(loads)
        projected_loads[edge_name] = projected_load
        direct_time = edge.travel_time(projected_load)
        direct_cost = edge.decision_cost(projected_load)
        remaining_cost = shortest_remaining_cost(
            edge.end, edges, graph, projected_loads, vehicle_weight
        )
        options.append(
            {
                "edge": edge_name,
                "from": edge.start,
                "to": edge.end,
                "projected_load_vehicles": round(projected_load, 4),
                "direct_time": round(direct_time, 4),
                "direct_toll": round(edge.toll, 4),
                "direct_decision_cost": round(direct_cost, 4),
                "estimated_remaining_cost": round(remaining_cost, 4),
                "estimated_total_cost": round(direct_cost + remaining_cost, 4),
            }
        )
    return options


def choose_edge(
    node: str,
    edges: dict[str, Edge],
    graph: dict[str, list[str]],
    loads: dict[str, float],
    vehicle_weight: float,
    decision_mode: str,
    driver: Optional[DriverAgent],
    stats: Counter,
) -> tuple[str, Decision]:
    options = make_options(node, edges, graph, loads, vehicle_weight)
    best_edge = min(options, key=lambda item: (item["estimated_total_cost"], item["edge"]))["edge"]

    if len(options) == 1:
        choice = options[0]["edge"]
        return choice, Decision(node, options, best_edge, choice, choice, "forced_single_edge", "")

    if decision_mode == "heuristic":
        return best_edge, Decision(node, options, best_edge, best_edge, best_edge, "heuristic", "")

    if driver is None:
        raise RuntimeError("A Langroid driver is required for an LLM decision mode.")

    stats["llm_calls"] += 1
    llm_choice, raw = driver.choose(node, options)
    if llm_choice is None:
        stats["invalid_llm_outputs"] += 1
        return best_edge, Decision(node, options, best_edge, None, best_edge, "invalid_llm_fallback", raw)
    if decision_mode == "llm_checked" and llm_choice != best_edge:
        stats["corrected_non_minimal_choices"] += 1
        return best_edge, Decision(node, options, best_edge, llm_choice, best_edge, "llm_corrected", raw)
    if llm_choice != best_edge:
        stats["accepted_non_minimal_choices"] += 1
    return llm_choice, Decision(node, options, best_edge, llm_choice, llm_choice, "llm_choice", raw)


def run_scenario(
    name: str,
    extended: bool,
    agents: int,
    vehicles: int,
    decision_mode: str,
    model: str,
    ollama_url: str,
    toll_ab: float = 0.0,
) -> dict:
    edges, graph = build_network(extended, toll_ab)
    vehicle_weight = vehicles / agents
    loads = {edge_name: 0.0 for edge_name in edges}
    stats: Counter = Counter()
    results: list[AgentResult] = []

    for agent_id in range(1, agents + 1):
        driver = None if decision_mode == "heuristic" else DriverAgent(model, ollama_url)
        node = "Start"
        route_nodes = [node]
        route_edges: list[str] = []
        decisions: list[Decision] = []
        while node != "Ziel":
            edge_name, decision = choose_edge(
                node, edges, graph, loads, vehicle_weight, decision_mode, driver, stats
            )
            edge = edges[edge_name]
            loads[edge_name] += vehicle_weight
            route_edges.append(edge_name)
            decisions.append(decision)
            node = edge.end
            route_nodes.append(node)
        total_toll = sum(edges[edge_name].toll for edge_name in route_edges)
        results.append(AgentResult(agent_id, route_nodes, route_edges, 0.0, total_toll, decisions))

    for result in results:
        result.final_time = round(sum(edges[edge].travel_time(loads[edge]) for edge in result.route_edges), 4)

    average_time = sum(result.final_time for result in results) / len(results)
    average_toll = sum(result.total_toll for result in results) / len(results)
    return {
        "name": name,
        "extended": extended,
        "toll_ab": toll_ab,
        "vehicle_weight": vehicle_weight,
        "loads": loads,
        "results": results,
        "stats": dict(stats),
        "average_time": average_time,
        "average_toll": average_toll,
        "average_decision_cost": average_time + average_toll,
    }


def write_results(output_dir: Path, scenario: dict, filename: str) -> None:
    filename = output_dir / filename
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["agent_id", "represented_vehicles", "route_nodes", "route_edges", "final_time", "total_toll", "decision_cost", "decisions_json"])
        for result in scenario["results"]:
            writer.writerow(
                [
                    result.agent_id,
                    scenario["vehicle_weight"],
                    " -> ".join(result.route_nodes),
                    " -> ".join(result.route_edges),
                    result.final_time,
                    result.total_toll,
                    result.final_time + result.total_toll,
                    json.dumps([asdict(decision) for decision in result.decisions]),
                ]
            )


def write_comparison(output_dir: Path, scenarios: list[tuple[str, Optional[float], dict]]) -> None:
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["scenario", "toll_ab", "reference_average_time", "simulated_average_time", "average_toll", "average_decision_cost", "difference", "route_distribution", "edge_loads", "stats"])
        for name, reference, result in scenarios:
            route_distribution = Counter(" -> ".join(agent.route_nodes) for agent in result["results"])
            writer.writerow(
                [
                    name,
                    result["toll_ab"],
                    reference,
                    f"{result['average_time']:.4f}",
                    f"{result['average_toll']:.4f}",
                    f"{result['average_decision_cost']:.4f}",
                    f"{result['average_time'] - reference:.4f}" if reference is not None else "",
                    json.dumps(dict(route_distribution)),
                    json.dumps(result["loads"]),
                    json.dumps(result["stats"]),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Braess traffic simulation using Langroid driver agents")
    parser.add_argument("--agents", type=int, default=500)
    parser.add_argument("--vehicles", type=int, default=4000)
    parser.add_argument("--decision-mode", choices=["heuristic", "llm_raw", "llm_checked"], default="heuristic")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1")
    parser.add_argument("--output-dir", type=Path, default=Path("results_langroid"))
    parser.add_argument("--tolls", type=float, nargs="*", default=[20.0, 30.0, 40.0, 50.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.agents <= 0 or args.vehicles <= 0:
        raise ValueError("--agents and --vehicles must be positive.")

    original = run_scenario(
        "Original network", False, args.agents, args.vehicles, args.decision_mode, args.model, args.ollama_url
    )
    extended = run_scenario(
        "Extended network with A-B link", True, args.agents, args.vehicles, args.decision_mode, args.model, args.ollama_url
    )
    toll_scenarios = [
        run_scenario(
            f"Extended network with A-B toll {toll:g}",
            True,
            args.agents,
            args.vehicles,
            args.decision_mode,
            args.model,
            args.ollama_url,
            toll,
        )
        for toll in args.tolls
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir, original, "original.csv")
    write_results(args.output_dir, extended, "extended.csv")
    for scenario in toll_scenarios:
        write_results(args.output_dir, scenario, f"extended_toll_{scenario['toll_ab']:g}.csv")
    reference_times = {20.0: 69.8676, 30.0: 66.1078, 40.0: 64.8948, 50.0: 65.0003}
    write_comparison(
        args.output_dir,
        [("original", 65.0, original), ("extended", 82.0, extended)]
        + [(f"extended_toll_{scenario['toll_ab']:g}", reference_times.get(scenario["toll_ab"]), scenario) for scenario in toll_scenarios],
    )

    print(f"Original network: {original['average_time']:.4f} min (reference 65.0000)")
    print(f"Extended network: {extended['average_time']:.4f} min (reference 82.0000)")
    for scenario in toll_scenarios:
        print(f"A-B toll {scenario['toll_ab']:g}: {scenario['average_time']:.4f} min")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()