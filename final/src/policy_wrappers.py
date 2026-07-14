"""Wrappers around Overcooked-AI Agent objects."""

from __future__ import annotations


import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

from src.constants import ACTION_NAME_TO_INDEX, NUM_ACTIONS, action_index_to_overcooked_action, action_name_to_index


@dataclass
class ActionResult:
    action: Any
    info: dict[str, Any]


class StudentAgentAdapter(Agent):
    """Adapt a student policy with act(obs)->int to Overcooked-AI's Agent API."""

    def __init__(self, student_agent, obs_builder, name: str = "student_agent"):
        super().__init__()
        self.student_agent = student_agent
        self.obs_builder = obs_builder
        self.name = name

    def reset(self):
        super().reset()
        if hasattr(self, "student_agent") and hasattr(self.student_agent, "reset"):
            self.student_agent.reset()

    def action(self, state):
        obs = self.obs_builder(state, self.agent_index)
        raw_action = self.student_agent.act(obs)
        action_index = coerce_action_index(raw_action)
        action = action_index_to_overcooked_action(action_index)
        return action, {"policy_name": self.name, "action_index": action_index}


class EpsilonActionWrapper(Agent):
    """Apply reproducible sticky actions followed by random action replacement.

    The historical class name is kept for backwards compatibility.  The order
    is deliberately:

    1. repeat the previous final action when sticky noise fires;
    2. otherwise query the wrapped policy;
    3. optionally replace that result with a uniform random action;
    4. remember the final action for the next timestep.
    """

    def __init__(
        self,
        base_agent: Agent,
        random_action_prob: float = 0.0,
        sticky_action_prob: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.base_agent = base_agent
        self.random_action_prob = float(random_action_prob)
        self.sticky_action_prob = float(sticky_action_prob)
        if not 0.0 <= self.random_action_prob <= 1.0:
            raise ValueError("random_action_prob must be in [0, 1]")
        if not 0.0 <= self.sticky_action_prob <= 1.0:
            raise ValueError("sticky_action_prob must be in [0, 1]")
        self.rng = np.random.default_rng(seed)
        self.last_action = None

    def reset(self):
        super().reset()
        self.last_action = None
        if hasattr(self, "base_agent"):
            self.base_agent.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.base_agent.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.base_agent.set_mdp(mdp)

    def action(self, state):
        sticky_override = False
        if self.last_action is not None and self.sticky_action_prob > 0:
            sticky_override = bool(self.rng.random() < self.sticky_action_prob)

        if sticky_override:
            action = self.last_action
            info = {"sticky_override": True}
        else:
            action, info = self.base_agent.action(state)
            info = dict(info or {})
            info["sticky_override"] = False

        if self.random_action_prob > 0 and self.rng.random() < self.random_action_prob:
            idx = int(self.rng.integers(0, NUM_ACTIONS))
            action = action_index_to_overcooked_action(idx)
            info["random_override"] = True
            info["random_action_index"] = idx
        else:
            info["random_override"] = False

        self.last_action = action
        return action, info


class SafeActionWrapper(Agent):
    """Handle invalid actions and slow policies.

    If the wrapped policy returns an invalid action or exceeds the time limit,
    this wrapper substitutes a configured fallback action.
    """

    def __init__(
        self,
        base_agent: Agent,
        max_action_time_ms: int | None = 100,
        invalid_action: str | int = "stay",
        timeout_action: str | int = "stay",
    ):
        super().__init__()
        self.base_agent = base_agent
        self.max_action_time_ms = None if max_action_time_ms is None else int(max_action_time_ms)
        self.invalid_action_idx = coerce_action_index(invalid_action)
        self.timeout_action_idx = coerce_action_index(timeout_action)
        self.invalid_count = 0
        self.timeout_count = 0

    def reset(self):
        super().reset()
        self.invalid_count = 0
        self.timeout_count = 0
        if hasattr(self, "base_agent"):
            self.base_agent.reset()

    def set_agent_index(self, agent_index):
        super().set_agent_index(agent_index)
        self.base_agent.set_agent_index(agent_index)

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.base_agent.set_mdp(mdp)

    def _call_base_agent(self, state) -> ActionResult:
        action, info = self.base_agent.action(state)
        if action not in Action.ALL_ACTIONS:
            raise ValueError(f"Invalid Overcooked action returned: {action}")
        return ActionResult(action=action, info=dict(info or {}))

    def action(self, state):
        start = time.perf_counter()
        try:
            if self.max_action_time_ms is None or self.max_action_time_ms <= 0:
                result = self._call_base_agent(state)
            else:
                with time_limit(self.max_action_time_ms / 1000.0):
                    result = self._call_base_agent(state)
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            # SIGALRM is unavailable on platforms such as Windows.  A call
            # cannot be interrupted there, but a policy that eventually
            # returns late must still be counted and replaced by the official
            # timeout fallback.
            if self.max_action_time_ms is not None and self.max_action_time_ms > 0:
                if elapsed_ms > self.max_action_time_ms:
                    self.timeout_count += 1
                    action = action_index_to_overcooked_action(self.timeout_action_idx)
                    return action, {
                        "elapsed_ms": elapsed_ms,
                        "timeout_action_replaced": True,
                        "timeout_count": self.timeout_count,
                    }
            result.info["elapsed_ms"] = elapsed_ms
            result.info["invalid_action_replaced"] = False
            result.info["timeout_action_replaced"] = False
            return result.action, result.info
        except TimeoutError:
            self.timeout_count += 1
            action = action_index_to_overcooked_action(self.timeout_action_idx)
            return action, {
                "elapsed_ms": 1000.0 * (time.perf_counter() - start),
                "timeout_action_replaced": True,
                "timeout_count": self.timeout_count,
            }
        except Exception as exc:
            self.invalid_count += 1
            action = action_index_to_overcooked_action(self.invalid_action_idx)
            return action, {
                "invalid_action_replaced": True,
                "invalid_count": self.invalid_count,
                "error": repr(exc),
            }


class time_limit:
    """Unix soft timeout using SIGALRM.

    This works in the main thread on Linux/macOS. If SIGALRM is unavailable,
    the code still runs, but timeout enforcement becomes best-effort.
    """

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.previous_handler = None

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(f"Policy exceeded {self.seconds:.3f} seconds")

    def __enter__(self):
        if hasattr(signal, "SIGALRM") and self.seconds > 0:
            self.previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if self.previous_handler is not None:
                signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def coerce_action_index(action_like: str | int) -> int:
    """Normalize action returned by a policy into an integer index."""
    if isinstance(action_like, str):
        return action_name_to_index(action_like)
    if isinstance(action_like, np.integer):
        action_like = int(action_like)
    if not isinstance(action_like, int):
        raise TypeError(f"Action must be int or action name, got {type(action_like)}")
    if action_like < 0 or action_like >= NUM_ACTIONS:
        raise ValueError(f"Action must be in [0, {NUM_ACTIONS - 1}], got {action_like}")
    return int(action_like)


def wrap_agent(base_agent: Agent, config: dict[str, Any], seed: int | None = None) -> Agent:
    """Apply safety, sticky-action and random-action wrappers from YAML."""
    safe = SafeActionWrapper(
        base_agent,
        max_action_time_ms=config.get("max_action_time_ms", 100),
        invalid_action=config.get("invalid_action", "stay"),
        timeout_action=config.get("timeout_action", "stay"),
    )
    epsilon = float(config.get("random_action_prob", 0.0) or 0.0)
    sticky = float(config.get("sticky_action_prob", 0.0) or 0.0)
    if epsilon > 0 or sticky > 0:
        return EpsilonActionWrapper(
            safe,
            random_action_prob=epsilon,
            sticky_action_prob=sticky,
            seed=seed,
        )
    return safe
