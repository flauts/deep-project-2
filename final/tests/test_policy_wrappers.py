import pytest
import time


pytest.importorskip("overcooked_ai_py")

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

from src.constants import action_index_to_overcooked_action, overcooked_action_to_index
from src.policy_wrappers import EpsilonActionWrapper, SafeActionWrapper


class CountingAgent(Agent):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def reset(self):
        super().reset()
        self.calls = 0

    def action(self, state):
        action = action_index_to_overcooked_action(self.calls % 6)
        self.calls += 1
        return action, {}


def _sequence(seed):
    wrapper = EpsilonActionWrapper(
        CountingAgent(),
        random_action_prob=0.20,
        sticky_action_prob=0.45,
        seed=seed,
    )
    wrapper.reset()
    return [overcooked_action_to_index(wrapper.action(None)[0]) for _ in range(30)]


def test_sticky_and_random_noise_are_reproducible_with_fixed_seed():
    assert _sequence(12345) == _sequence(12345)


def test_sticky_repeats_previous_final_action_and_resets():
    base = CountingAgent()
    wrapper = EpsilonActionWrapper(base, sticky_action_prob=1.0, seed=7)
    wrapper.reset()

    first = wrapper.action(None)[0]
    second = wrapper.action(None)[0]
    assert second == first
    assert base.calls == 1

    wrapper.reset()
    wrapper.action(None)
    assert base.calls == 1


class SlowAgent(Agent):
    def action(self, state):
        time.sleep(0.01)
        return Action.INTERACT, {}


def test_late_return_is_counted_as_timeout_and_replaced():
    wrapper = SafeActionWrapper(SlowAgent(), max_action_time_ms=1, timeout_action="stay")

    action, info = wrapper.action(None)

    assert action == Action.STAY
    assert wrapper.timeout_count == 1
    assert info["timeout_action_replaced"] is True
