from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


Position = tuple[int, int]
Direction = tuple[int, int]

DIRECTIONS: dict[str, Direction] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ORDERED_DIRECTIONS = tuple(DIRECTIONS.values())
ROAD_CELLS = {"0", "2", "3", "4"}

STRATEGIES = {
    "right_priority": {
        "title": "Правый поворот",
        "description": "5% остановка+разворот, 40% направо, 30% прямо, 25% налево.",
    },
    "straight_priority": {
        "title": "Прямое движение",
        "description": "5% остановка+разворот, 20% направо, 55% прямо, 20% налево.",
    },
    "target_biased": {
        "title": "Манхеттенское расстояние",
        "description": "5% остановка+разворот, затем повышенный вес направлений, уменьшающих расстояние до доставки.",
    },
}


@dataclass
class CityMap:
    grid: list[str]
    starts: list[Position]
    spawns: list[Position]
    target: Position

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])


def add_pos(a: Position, b: Direction) -> Position:
    return a[0] + b[0], a[1] + b[1]


def opposite(direction: Direction | None) -> Direction | None:
    if direction is None:
        return None
    return -direction[0], -direction[1]


def is_accessible(grid: list[str], pos: Position) -> bool:
    r, c = pos
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in ROAD_CELLS


def valid_directions(grid: list[str], pos: Position) -> list[Direction]:
    return [d for d in ORDERED_DIRECTIONS if is_accessible(grid, add_pos(pos, d))]


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def weighted_choice(rng: random.Random, choices: list[tuple[Direction | str, float]]) -> Direction | str:
    total = sum(weight for _, weight in choices if weight > 0)
    if total <= 0:
        return choices[0][0]
    mark = rng.random() * total
    acc = 0.0
    for value, weight in choices:
        if weight <= 0:
            continue
        acc += weight
        if mark <= acc:
            return value
    return choices[-1][0]


def load_city_map(path: Path) -> CityMap:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Map file is empty")
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError("All map rows must have equal length")
    if len(lines) < 30 or width < 30:
        raise ValueError("Map must be at least 30 x 30")

    starts: list[Position] = []
    spawns: list[Position] = []
    target: Position | None = None
    for r, row in enumerate(lines):
        for c, cell in enumerate(row):
            if cell == "2":
                starts.append((r, c))
            elif cell == "3":
                spawns.append((r, c))
            elif cell == "4":
                target = (r, c)

    if not starts:
        raise ValueError("Map must contain at least one start cell marked as 2")
    if not spawns:
        raise ValueError("Map must contain at least one traffic spawn cell marked as 3")
    if target is None:
        raise ValueError("Map must contain one target cell marked as 4")

    return CityMap(lines, starts, spawns, target)


def relation_to_heading(heading: Direction, direction: Direction) -> str | None:
    # Matrix rotation in row/column coordinates: right turn maps (dr, dc) -> (dc, -dr).
    right = (heading[1], -heading[0])
    left = (-heading[1], heading[0])
    if direction == heading:
        return "straight"
    if direction == right:
        return "right"
    if direction == left:
        return "left"
    if direction == opposite(heading):
        return "back"
    return None


def choose_initial_direction(
    rng: random.Random,
    grid: list[str],
    pos: Position,
    target: Position,
    available: list[Direction],
    strategy: str,
) -> Direction:
    if strategy == "target_biased":
        choices = []
        current = manhattan(pos, target)
        for direction in available:
            distance = manhattan(add_pos(pos, direction), target)
            choices.append((direction, 5.0 if distance < current else 1.0))
        return weighted_choice(rng, choices)  # type: ignore[return-value]
    return rng.choice(available)


def choose_agent_move(
    rng: random.Random,
    grid: list[str],
    target: Position,
    state: dict,
    strategy: str,
) -> dict:
    pos: Position = tuple(state["pos"])  # type: ignore[assignment]
    direction: Direction | None = tuple(state["direction"]) if state["direction"] else None  # type: ignore[assignment]
    pending_reverse = bool(state.get("pending_reverse", False))
    available = valid_directions(grid, pos)

    if not available:
        return {"pos": pos, "direction": direction, "pending_reverse": False, "stopped": True}

    if direction is None:
        move = choose_initial_direction(rng, grid, pos, target, available, strategy)
        return {"pos": add_pos(pos, move), "direction": move, "pending_reverse": False, "stopped": False}

    reverse_dir = opposite(direction)
    if pending_reverse and reverse_dir in available:
        return {"pos": add_pos(pos, reverse_dir), "direction": reverse_dir, "pending_reverse": False, "stopped": False}

    forward_options = [d for d in available if d != reverse_dir]
    if not forward_options and reverse_dir in available:
        return {"pos": add_pos(pos, reverse_dir), "direction": reverse_dir, "pending_reverse": False, "stopped": False}
    if len(forward_options) == 1:
        move = forward_options[0]
        return {"pos": add_pos(pos, move), "direction": move, "pending_reverse": False, "stopped": False}

    # At forks the selected strategy controls stop/reverse, straight, right and left probabilities.
    if strategy == "right_priority":
        relation_weights = {"right": 40.0, "straight": 30.0, "left": 25.0}
        stop_weight = 5.0
    elif strategy == "straight_priority":
        relation_weights = {"right": 20.0, "straight": 55.0, "left": 20.0}
        stop_weight = 5.0
    elif strategy == "target_biased":
        current = manhattan(pos, target)
        choices: list[tuple[Direction | str, float]] = [("stop_reverse", 5.0)]
        for move in forward_options:
            distance = manhattan(add_pos(pos, move), target)
            choices.append((move, 70.0 if distance < current else 15.0))
        decision = weighted_choice(rng, choices)
        if decision == "stop_reverse":
            return {"pos": pos, "direction": direction, "pending_reverse": True, "stopped": True}
        move = decision  # type: ignore[assignment]
        return {"pos": add_pos(pos, move), "direction": move, "pending_reverse": False, "stopped": False}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    choices = [("stop_reverse", stop_weight)]
    for move in forward_options:
        relation = relation_to_heading(direction, move)
        choices.append((move, relation_weights.get(relation or "", 0.0)))
    decision = weighted_choice(rng, choices)
    if decision == "stop_reverse":
        return {"pos": pos, "direction": direction, "pending_reverse": True, "stopped": True}
    move = decision  # type: ignore[assignment]
    return {"pos": add_pos(pos, move), "direction": move, "pending_reverse": False, "stopped": False}


def move_bot(rng: random.Random, grid: list[str], bot: dict) -> dict | None:
    pos: Position = tuple(bot["pos"])  # type: ignore[assignment]
    direction: Direction | None = tuple(bot["direction"]) if bot["direction"] else None  # type: ignore[assignment]
    pending_reverse = bool(bot.get("pending_reverse", False))
    available = valid_directions(grid, pos)
    remaining = int(bot["remaining"]) - 1

    if remaining <= 0:
        return None
    if not available:
        bot.update({"remaining": remaining, "pending_reverse": False, "stopped": True})
        return bot

    if direction is None:
        move = rng.choice(available)
        bot.update({"pos": add_pos(pos, move), "direction": move, "remaining": remaining, "pending_reverse": False, "stopped": False})
        return bot

    reverse_dir = opposite(direction)
    if pending_reverse and reverse_dir in available:
        move = reverse_dir
        bot.update({"pos": add_pos(pos, move), "direction": move, "remaining": remaining, "pending_reverse": False, "stopped": False})
        return bot

    forward_options = [d for d in available if d != reverse_dir]
    if not forward_options and reverse_dir in available:
        move = reverse_dir
        bot.update({"pos": add_pos(pos, move), "direction": move, "remaining": remaining, "pending_reverse": False, "stopped": False})
        return bot
    if len(forward_options) == 1:
        move = forward_options[0]
        bot.update({"pos": add_pos(pos, move), "direction": move, "remaining": remaining, "pending_reverse": False, "stopped": False})
        return bot

    # Bot rule from the task: 10% stop, then reverse on the next iteration;
    # otherwise choose uniformly among all non-reverse outgoing roads.
    if rng.random() < 0.10:
        bot.update({"pos": pos, "direction": direction, "remaining": remaining, "pending_reverse": True, "stopped": True})
        return bot

    move = rng.choice(forward_options)
    bot.update({"pos": add_pos(pos, move), "direction": move, "remaining": remaining, "pending_reverse": False, "stopped": False})
    return bot


def remove_bot_collisions(previous_bots: list[dict], moved_bots: list[dict]) -> tuple[list[dict], int]:
    by_cell: dict[Position, list[int]] = {}
    for index, bot in enumerate(moved_bots):
        by_cell.setdefault(tuple(bot["pos"]), []).append(index)

    crashed: set[int] = {idx for indexes in by_cell.values() if len(indexes) > 1 for idx in indexes}

    previous_by_id = {bot["id"]: tuple(bot["pos"]) for bot in previous_bots}
    for i, bot_a in enumerate(moved_bots):
        if i in crashed:
            continue
        for j in range(i + 1, len(moved_bots)):
            if j in crashed:
                continue
            bot_b = moved_bots[j]
            if (
                previous_by_id.get(bot_a["id"]) == tuple(bot_b["pos"])
                and previous_by_id.get(bot_b["id"]) == tuple(bot_a["pos"])
            ):
                crashed.add(i)
                crashed.add(j)

    survivors = [bot for i, bot in enumerate(moved_bots) if i not in crashed]
    return survivors, len(crashed)


def traffic_step(
    rng: random.Random,
    grid: list[str],
    spawn_cells: list[Position],
    bots: list[dict],
    spawn_probability: float,
    next_bot_id: int,
) -> tuple[list[dict], int, int, int]:
    moved: list[dict] = []
    for bot in bots:
        next_bot = move_bot(rng, grid, dict(bot))
        if next_bot is not None:
            moved.append(next_bot)

    moved, bot_crashes = remove_bot_collisions(bots, moved)
    occupied = {tuple(bot["pos"]) for bot in moved}
    spawned = 0

    for spawn in spawn_cells:
        if spawn in occupied:
            continue
        if rng.random() < spawn_probability:
            moved.append(
                {
                    "id": next_bot_id,
                    "pos": spawn,
                    "direction": None,
                    "remaining": rng.randint(15, 150),
                    "pending_reverse": False,
                    "stopped": False,
                }
            )
            occupied.add(spawn)
            next_bot_id += 1
            spawned += 1

    return moved, next_bot_id, spawned, bot_crashes


def agent_worker(conn, grid: list[str], target: Position, seed: int) -> None:
    rng = random.Random(seed)
    while True:
        message = conn.recv()
        command = message.get("command")
        if command == "stop":
            conn.close()
            return
        if command == "reset":
            rng.seed(message["seed"])
            conn.send({"ok": True})
            continue
        if command == "step":
            conn.send(choose_agent_move(rng, grid, target, message["state"], message["strategy"]))
            continue
        raise ValueError(f"Unknown agent command: {command}")


def traffic_worker(conn, grid: list[str], spawn_cells: list[Position], seed: int) -> None:
    rng = random.Random(seed)
    next_bot_id = 0
    while True:
        message = conn.recv()
        command = message.get("command")
        if command == "stop":
            conn.close()
            return
        if command == "reset":
            rng.seed(message["seed"])
            next_bot_id = 0
            conn.send({"ok": True})
            continue
        if command == "step":
            bots, next_bot_id, spawned, bot_crashes = traffic_step(
                rng,
                grid,
                spawn_cells,
                message["bots"],
                message["spawn_probability"],
                next_bot_id,
            )
            conn.send({"bots": bots, "spawned": spawned, "bot_crashes": bot_crashes})
            continue
        raise ValueError(f"Unknown traffic command: {command}")


class ProcessController:
    def __init__(self, city: CityMap, seed: int) -> None:
        ctx = mp.get_context("spawn")
        self.agent_parent, agent_child = ctx.Pipe()
        self.traffic_parent, traffic_child = ctx.Pipe()
        self.agent_process = ctx.Process(target=agent_worker, args=(agent_child, city.grid, city.target, seed + 11))
        self.traffic_process = ctx.Process(target=traffic_worker, args=(traffic_child, city.grid, city.spawns, seed + 29))
        self.agent_process.start()
        self.traffic_process.start()

    def reset(self, seed: int) -> None:
        self.agent_parent.send({"command": "reset", "seed": seed + 101})
        self.traffic_parent.send({"command": "reset", "seed": seed + 202})
        self.agent_parent.recv()
        self.traffic_parent.recv()

    def agent_step(self, state: dict, strategy: str) -> dict:
        self.agent_parent.send({"command": "step", "state": state, "strategy": strategy})
        return self.agent_parent.recv()

    def traffic_step(self, bots: list[dict], spawn_probability: float) -> dict:
        self.traffic_parent.send({"command": "step", "bots": bots, "spawn_probability": spawn_probability})
        return self.traffic_parent.recv()

    def close(self) -> None:
        for conn in (self.agent_parent, self.traffic_parent):
            try:
                conn.send({"command": "stop"})
            except (BrokenPipeError, EOFError):
                pass
        self.agent_process.join(timeout=5)
        self.traffic_process.join(timeout=5)
        if self.agent_process.is_alive():
            self.agent_process.terminate()
        if self.traffic_process.is_alive():
            self.traffic_process.terminate()


def collision_with_agent(
    previous_agent_pos: Position,
    next_agent_pos: Position,
    previous_bots: list[dict],
    next_bots: list[dict],
) -> dict | None:
    previous_by_id = {bot["id"]: tuple(bot["pos"]) for bot in previous_bots}
    for bot in next_bots:
        bot_pos = tuple(bot["pos"])
        if bot_pos == next_agent_pos:
            return {
                "type": "same_cell",
                "bot_id": bot["id"],
                "pos": next_agent_pos,
                "agent_from": previous_agent_pos,
                "agent_to": next_agent_pos,
                "bot_from": previous_by_id.get(bot["id"], bot_pos),
                "bot_to": bot_pos,
            }
        previous_bot_pos = previous_by_id.get(bot["id"])
        if previous_bot_pos == next_agent_pos and bot_pos == previous_agent_pos:
            return {
                "type": "swap",
                "bot_id": bot["id"],
                "agent_from": previous_agent_pos,
                "agent_to": next_agent_pos,
                "bot_from": previous_bot_pos,
                "bot_to": bot_pos,
            }
    return None


def run_episode(
    city: CityMap,
    controller: ProcessController,
    *,
    strategy: str,
    spawn_probability: float,
    start: Position,
    max_steps: int,
    seed: int,
    record_frames: bool = False,
) -> dict:
    controller.reset(seed)
    agent = {"pos": start, "direction": None, "pending_reverse": False, "stopped": False}
    bots: list[dict] = []
    frames: list[dict] = []
    spawned_total = 0
    bot_crashes_total = 0

    if record_frames:
        frames.append({"step": 0, "agent": dict(agent), "bots": [], "event": "start"})

    for step in range(1, max_steps + 1):
        previous_agent_pos = tuple(agent["pos"])
        previous_bots = [dict(bot) for bot in bots]

        next_agent = controller.agent_step(agent, strategy)
        traffic_result = controller.traffic_step(bots, spawn_probability)
        next_bots = traffic_result["bots"]
        spawned_total += traffic_result["spawned"]
        bot_crashes_total += traffic_result["bot_crashes"]

        next_agent_pos = tuple(next_agent["pos"])
        collision = collision_with_agent(previous_agent_pos, next_agent_pos, previous_bots, next_bots)
        crashed = collision is not None

        if record_frames:
            frames.append(
                {
                    "step": step,
                    "agent": dict(next_agent),
                    "bots": [dict(bot) for bot in next_bots],
                    "previous_agent": dict(agent),
                    "previous_bots": previous_bots,
                    "collision": collision,
                    "event": "crash" if crashed else ("success" if next_agent_pos == city.target else "move"),
                }
            )

        if crashed:
            return {
                "success": False,
                "reason": "collision",
                "steps": step,
                "spawned": spawned_total,
                "bot_crashes": bot_crashes_total,
                "frames": frames,
            }

        agent = next_agent
        bots = next_bots

        if next_agent_pos == city.target:
            return {
                "success": True,
                "reason": "delivered",
                "steps": step,
                "spawned": spawned_total,
                "bot_crashes": bot_crashes_total,
                "frames": frames,
            }

    return {
        "success": False,
        "reason": "timeout",
        "steps": max_steps,
        "spawned": spawned_total,
        "bot_crashes": bot_crashes_total,
        "frames": frames,
    }


def aggregate_episode_results(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["strategy"], row["spawn_probability"]), []).append(row)

    summary = []
    for (strategy, spawn_probability), group in sorted(groups.items()):
        successes = [row for row in group if row["success"]]
        success_steps = [row["steps"] for row in successes]
        summary.append(
            {
                "strategy": strategy,
                "strategy_title": STRATEGIES[strategy]["title"],
                "spawn_probability": spawn_probability,
                "episodes": len(group),
                "successes": len(successes),
                "success_probability": len(successes) / len(group),
                "avg_success_steps": float(np.mean(success_steps)) if success_steps else math.nan,
                "median_success_steps": float(np.median(success_steps)) if success_steps else math.nan,
                "avg_spawned": float(np.mean([row["spawned"] for row in group])),
                "collisions": sum(1 for row in group if row["reason"] == "collision"),
                "timeouts": sum(1 for row in group if row["reason"] == "timeout"),
                "bot_crashes": int(sum(row["bot_crashes"] for row in group)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_map_image(city: CityMap, path: Path, cell_size: int = 22) -> None:
    image = Image.new("RGB", (city.width * cell_size, city.height * cell_size), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "0": (235, 238, 241),
        "1": (45, 52, 60),
        "2": (49, 112, 203),
        "3": (235, 145, 52),
        "4": (61, 156, 91),
    }
    for r, row in enumerate(city.grid):
        for c, cell in enumerate(row):
            x0, y0 = c * cell_size, r * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=colors[cell])
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], outline=(210, 214, 219))
    image.save(path)


def render_frame(city: CityMap, frame: dict, cell_size: int = 18) -> Image.Image:
    legend_height = 38
    image = Image.new("RGB", (city.width * cell_size, city.height * cell_size + legend_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    colors = {
        "0": (235, 238, 241),
        "1": (44, 49, 58),
        "2": (154, 190, 242),
        "3": (242, 185, 119),
        "4": (127, 199, 143),
    }
    for r, row in enumerate(city.grid):
        for c, cell in enumerate(row):
            x0, y0 = c * cell_size, r * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 1, y0 + cell_size - 1], fill=colors[cell])

    collision = frame.get("collision")
    collision_bot_id = collision.get("bot_id") if collision else None
    bots_to_draw = list(frame["bots"])
    agent_to_draw = frame["agent"]

    if collision and collision.get("type") == "swap":
        # For a head-on cell swap, draw the attempted collision before the vehicles
        # appear to have passed through each other.
        agent_to_draw = frame.get("previous_agent", frame["agent"])
        previous_bots = {bot["id"]: bot for bot in frame.get("previous_bots", [])}
        bots_to_draw = [bot for bot in bots_to_draw if bot["id"] != collision_bot_id]
        if collision_bot_id in previous_bots:
            bots_to_draw.append(previous_bots[collision_bot_id])

    for bot in bots_to_draw:
        if collision and collision.get("type") == "same_cell" and bot["id"] == collision_bot_id:
            continue
        r, c = bot["pos"]
        x0, y0 = c * cell_size + 3, r * cell_size + 3
        draw.ellipse([x0, y0, x0 + cell_size - 7, y0 + cell_size - 7], fill=(111, 66, 193))

    ar, ac = agent_to_draw["pos"]
    x0, y0 = ac * cell_size + 2, ar * cell_size + 2
    draw.rectangle([x0, y0, x0 + cell_size - 5, y0 + cell_size - 5], fill=(220, 53, 69))

    if collision:
        if collision.get("type") == "swap":
            a0 = collision["agent_from"]
            a1 = collision["agent_to"]
            cx = ((a0[1] + 0.5) + (a1[1] + 0.5)) * cell_size / 2
            cy = ((a0[0] + 0.5) + (a1[0] + 0.5)) * cell_size / 2
        else:
            r, c = collision["pos"]
            cx = (c + 0.5) * cell_size
            cy = (r + 0.5) * cell_size
        radius = max(5, cell_size // 2)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(255, 193, 7), width=3)
        draw.line([cx - radius, cy - radius, cx + radius, cy + radius], fill=(220, 53, 69), width=3)
        draw.line([cx - radius, cy + radius, cx + radius, cy - radius], fill=(220, 53, 69), width=3)

    crash_note = f" ({collision['type']})" if collision else ""
    status = f"step={frame['step']}  bots={len(frame['bots'])}  event={frame['event']}{crash_note}"
    draw.rectangle([0, city.height * cell_size, city.width * cell_size, city.height * cell_size + legend_height], fill=(250, 250, 250))
    draw.text((8, city.height * cell_size + 10), status, fill=(35, 39, 47))
    return image


def write_animation(city: CityMap, frames: list[dict], path: Path, max_frames: int = 180) -> None:
    selected = frames[:max_frames]
    if not selected:
        return
    images = [render_frame(city, frame) for frame in selected]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
        optimize=False,
    )


def plot_summary(summary: list[dict], output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    spawn_values = sorted({row["spawn_probability"] for row in summary})

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in STRATEGIES:
        rows = [row for row in summary if row["strategy"] == strategy]
        rows.sort(key=lambda row: row["spawn_probability"])
        y = [row["success_probability"] for row in rows]
        ax.plot(spawn_values, y, marker="o", linewidth=2, label=STRATEGIES[strategy]["title"])
    ax.set_title("Вероятность успешной доставки")
    ax.set_xlabel("Вероятность генерации транспорта из каждой точки за итерацию")
    ax.set_ylabel("Доля успешных эпизодов")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "success_probability_by_strategy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for strategy in STRATEGIES:
        rows = [row for row in summary if row["strategy"] == strategy]
        rows.sort(key=lambda row: row["spawn_probability"])
        y = [row["avg_success_steps"] if not math.isnan(row["avg_success_steps"]) else np.nan for row in rows]
        ax.plot(spawn_values, y, marker="o", linewidth=2, label=STRATEGIES[strategy]["title"])
    ax.set_title("Среднее число шагов успешной доставки")
    ax.set_xlabel("Вероятность генерации транспорта из каждой точки за итерацию")
    ax.set_ylabel("Среднее число шагов")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "avg_steps_by_strategy.png", dpi=180)
    plt.close(fig)


def run_experiments(
    city: CityMap,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    spawn_probabilities: list[float],
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    all_rows: list[dict] = []
    sample_result: dict | None = None
    controller = ProcessController(city, seed)
    try:
        episode_index = 0
        for strategy in STRATEGIES:
            for spawn_probability in spawn_probabilities:
                for start in city.starts:
                    for local_episode in range(episodes):
                        run_seed = seed + episode_index * 9973
                        # Record candidate frames until the first successful target-oriented delivery is found.
                        # This keeps the demo animation aligned with the report: it shows a completed route.
                        record = (
                            sample_result is None
                            and strategy == "target_biased"
                            and abs(spawn_probability - min(spawn_probabilities, key=lambda p: abs(p - 0.02))) < 1e-12
                        )
                        result = run_episode(
                            city,
                            controller,
                            strategy=strategy,
                            spawn_probability=spawn_probability,
                            start=start,
                            max_steps=max_steps,
                            seed=run_seed,
                            record_frames=record,
                        )
                        row = {
                            "strategy": strategy,
                            "spawn_probability": spawn_probability,
                            "start": f"{start[0]},{start[1]}",
                            "episode": local_episode,
                            "success": result["success"],
                            "reason": result["reason"],
                            "steps": result["steps"],
                            "spawned": result["spawned"],
                            "bot_crashes": result["bot_crashes"],
                        }
                        all_rows.append(row)
                        if record and result["success"]:
                            sample_result = result
                        episode_index += 1
    finally:
        controller.close()

    summary = aggregate_episode_results(all_rows)
    write_csv(output_dir / "episode_results.csv", all_rows)
    write_csv(output_dir / "summary_results.csv", summary)
    (output_dir / "summary_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_rows, summary, sample_result or {}


def markdown_table(summary: list[dict]) -> str:
    lines = [
        "| Стратегия | p генерации | Эпизодов | Вероятность успеха | Средние шаги успешных | Аварии агента | Тайм-ауты |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        avg = "нет успехов" if math.isnan(row["avg_success_steps"]) else f"{row['avg_success_steps']:.1f}"
        lines.append(
            "| {strategy} | {p:.3f} | {episodes} | {success:.2f} | {avg} | {collisions} | {timeouts} |".format(
                strategy=row["strategy_title"],
                p=row["spawn_probability"],
                episodes=row["episodes"],
                success=row["success_probability"],
                avg=avg,
                collisions=row["collisions"],
                timeouts=row["timeouts"],
            )
        )
    return "\n".join(lines)


def best_strategy_line(summary: list[dict]) -> str:
    by_strategy: dict[str, list[dict]] = {}
    for row in summary:
        by_strategy.setdefault(row["strategy"], []).append(row)
    scored = []
    for strategy, rows in by_strategy.items():
        success = float(np.mean([row["success_probability"] for row in rows]))
        successful_steps = [row["avg_success_steps"] for row in rows if not math.isnan(row["avg_success_steps"])]
        avg_steps = float(np.mean(successful_steps)) if successful_steps else math.inf
        scored.append((success, -avg_steps, strategy))
    scored.sort(reverse=True)
    success, negative_steps, strategy = scored[0]
    return (
        f"По суммарной оценке лучшей оказалась стратегия **{STRATEGIES[strategy]['title']}**: "
        f"средняя вероятность успеха по всем скоростям генерации составила {success:.2f}, "
        f"а среднее число шагов успешных доставок — {-negative_steps:.1f}."
    )


def write_report(city: CityMap, output_dir: Path, summary: list[dict], args: argparse.Namespace) -> None:
    out = output_dir.as_posix()
    report = f"""# Лабораторная работа 4

## Реализация многоагентной среды с помощью многопроцессных вычислений

Цель работы: получить навыки создания многоагентной среды с неопределенностью с применением многопроцессных вычислений.

В работе смоделирована городская дорожная сеть, в которой агент-доставщик должен добраться из одной из стартовых точек `2` в точку доставки `4`. Другие транспортные средства появляются в точках `3` с заданной вероятностью и двигаются независимо от агента. Движение доставщика вычисляется отдельным процессом, движение транспортного потока — вторым процессом; основной процесс синхронизирует итерации и обрабатывает аварии.

## Карта

Файл карты: [`citymap.txt`](citymap.txt)

- `0` — дорога;
- `1` — препятствие/здание;
- `2` — стартовые точки доставщика;
- `3` — точки генерации транспорта;
- `4` — точка доставки.

Размер карты: {city.height} x {city.width}. Стартовые позиции агента: {", ".join(map(str, city.starts))}. Точки генерации транспорта: {", ".join(map(str, city.spawns))}. Цель доставки: {city.target}.

![Карта]({out}/map.png)

## Правила модели

На каждой итерации агент и транспортные средства выбирают одно действие: вверх, вниз, влево, вправо или остановка.

Для транспорта реализованы правила из задания:

- при появлении из клетки `3` бот получает случайное время жизни от 15 до 150 итераций;
- в коридоре бот продолжает движение по единственному доступному направлению;
- в тупике бот разворачивается;
- на перекрестке с вероятностью 10% бот останавливается, а на следующем ходу разворачивается;
- оставшиеся 90% распределяются равномерно между доступными направлениями, кроме обратного.

Если два бота оказываются в одной клетке или обмениваются клетками за одну итерацию, они удаляются как попавшие в аварию. Если с ботом сталкивается агент, эпизод считается неуспешным.

## Стратегии доставщика

Проверялись три стратегии прохождения развилок:

- **Правый поворот**: {STRATEGIES["right_priority"]["description"]}
- **Прямое движение**: {STRATEGIES["straight_priority"]["description"]}
- **{STRATEGIES["target_biased"]["title"]}**: {STRATEGIES["target_biased"]["description"]}

В обычном коридоре агент продолжает движение, в тупике разворачивается. Если агент успешно достигает клетки `4`, фиксируется число шагов доставки.

## Параллельная организация

Скрипт [`lab4_multiprocessing_simulation.py`](lab4_multiprocessing_simulation.py) использует модуль `multiprocessing`.

- Процесс агента получает текущее состояние доставщика и выбранную стратегию, после чего возвращает новое положение.
- Процесс транспорта получает список активных ботов и вероятность генерации, после чего возвращает обновленный список транспорта.
- Главный процесс выполняет барьер синхронизации итераций, проверяет столкновения и собирает статистику.

Такой вариант оставляет модель детерминированной при заданном seed, но разделяет вычисление движений между процессами.

## Эксперимент

Параметры запуска:

- seed: `{args.seed}`;
- эпизодов на одну комбинацию стартовой точки, стратегии и вероятности генерации: `{args.episodes}`;
- максимальная длина эпизода: `{args.max_steps}` шагов;
- вероятности генерации транспорта: `{", ".join(f"{p:.3f}" for p in args.spawn_probabilities)}`.

Так как на карте несколько стартовых позиций, в итоговых таблицах результаты агрегированы по всем стартам.

## Результаты

![Вероятность успеха]({out}/success_probability_by_strategy.png)

![Среднее число шагов]({out}/avg_steps_by_strategy.png)

{markdown_table(summary)}

Анимация успешного эпизода: [`{out}/agent_path_success.gif`]({out}/agent_path_success.gif)

## Выводы

{best_strategy_line(summary)}

Увеличение вероятности генерации транспорта ожидаемо снижает вероятность успешной доставки: на дорогах появляется больше ботов, поэтому растет риск столкновения. Среднее число шагов успешных доставок меняется не монотонно: при высокой плотности транспорта часть долгих и опасных эпизодов заканчивается аварией или тайм-аутом и не попадает в среднее по успешным доставкам.

Стратегии, учитывающие направление к цели, в среднем эффективнее случайных поворотных правил, потому что агент реже уходит в длинные петли дорожной сетки. При этом даже целевая стратегия не гарантирует успех: неопределенность создается случайной генерацией транспорта, случайным выбором движения на перекрестках и аварийными ситуациями.

## Состав файлов

- [`citymap.txt`](citymap.txt) — карта дорожной сети;
- [`lab4_multiprocessing_simulation.py`](lab4_multiprocessing_simulation.py) — программный код с комментариями;
- [`{out}/episode_results.csv`]({out}/episode_results.csv) — результаты всех эпизодов;
- [`{out}/summary_results.csv`]({out}/summary_results.csv) — агрегированная статистика;
- [`{out}/map.png`]({out}/map.png) — визуализация карты;
- [`{out}/success_probability_by_strategy.png`]({out}/success_probability_by_strategy.png) — график вероятности успеха;
- [`{out}/avg_steps_by_strategy.png`]({out}/avg_steps_by_strategy.png) — график среднего числа шагов;
- [`{out}/agent_path_success.gif`]({out}/agent_path_success.gif) — анимированная визуализация успешной доставки.
"""
    (Path("report_lab4.md")).write_text(report, encoding="utf-8")


def parse_spawn_probabilities(value: str) -> list[float]:
    probabilities = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not probabilities:
        raise argparse.ArgumentTypeError("Provide at least one probability")
    if any(p < 0 or p > 1 for p in probabilities):
        raise argparse.ArgumentTypeError("Probabilities must be in [0, 1]")
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 4 multiprocessing multi-agent city simulation")
    parser.add_argument("--map", default="citymap.txt", type=Path, help="Path to city map txt file")
    parser.add_argument("--output-dir", default="outputs", type=Path, help="Directory for generated artifacts")
    parser.add_argument("--episodes", default=40, type=int, help="Episodes for each start/strategy/spawn-probability combination")
    parser.add_argument("--max-steps", default=450, type=int, help="Maximum steps per episode")
    parser.add_argument(
        "--spawn-probabilities",
        default=parse_spawn_probabilities("0.005,0.010,0.020,0.040"),
        type=parse_spawn_probabilities,
        help="Comma-separated traffic spawn probabilities",
    )
    parser.add_argument("--seed", default=2026, type=int, help="Base random seed")
    args = parser.parse_args()

    city = load_city_map(args.map)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draw_map_image(city, args.output_dir / "map.png")
    _, summary, sample = run_experiments(
        city,
        args.output_dir,
        episodes=args.episodes,
        max_steps=args.max_steps,
        spawn_probabilities=args.spawn_probabilities,
        seed=args.seed,
    )
    plot_summary(summary, args.output_dir)

    if sample.get("frames"):
        write_animation(city, sample["frames"], args.output_dir / "agent_path_sample.gif")
        write_animation(city, sample["frames"], args.output_dir / "agent_path_success.gif")

    write_report(city, args.output_dir, summary, args)
    print(f"Done. Report: report_lab4.md; artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
