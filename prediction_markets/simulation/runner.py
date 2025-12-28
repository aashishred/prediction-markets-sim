"""
Simulation runner for prediction market experiments.

Orchestrates the interaction between environment, market, and agents
over multiple timesteps, collecting data for analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from ..agents.base import Agent
from ..environments.base import Environment
from ..markets.base import Market, Trade


@dataclass
class SimulationConfig:
    """
    Configuration for a simulation run.

    Attributes:
        n_steps: Number of timesteps to run
        agents_per_step: How many agents act per step (None = all)
        agent_order: Order of agent actions ('sequential', 'random', 'simultaneous')
        record_frequency: How often to record state (every N steps)
        random_seed: Random seed for reproducibility
        distribute_signals_once: Whether to distribute signals only at start
    """
    n_steps: int = 100
    agents_per_step: int | None = None
    agent_order: str = "random"
    record_frequency: int = 1
    random_seed: int | None = None
    distribute_signals_once: bool = True


@dataclass
class SimulationResult:
    """
    Results from a simulation run.

    Contains all data needed for analysis: price history, agent states,
    trades, and resolution outcomes.
    """
    config: SimulationConfig
    n_steps: int
    true_value: Any

    # Market data
    price_history: np.ndarray  # Shape: (n_steps, n_outcomes)
    quantity_history: np.ndarray  # Shape: (n_steps, n_outcomes)
    trades: list[Trade]
    final_prices: np.ndarray

    # Agent data
    agent_final_wealth: dict[str, float]
    agent_pnl: dict[str, float]
    agent_trades: dict[str, list[Trade]]

    # Resolution
    payouts: dict[str, float]
    total_subsidy_spent: float

    # Theoretical benchmarks
    theoretical_aggregate: np.ndarray | None = None
    prior: np.ndarray | None = None


@dataclass
class Simulation:
    """
    Main simulation class that runs prediction market experiments.

    Coordinates the environment, market, and agents through time,
    collecting data for later analysis.
    """
    environment: Environment
    market: Market
    agents: list[Agent]
    config: SimulationConfig = field(default_factory=SimulationConfig)

    rng: np.random.Generator = field(default=None, repr=False)
    _step: int = 0
    _trades: list[Trade] = field(default_factory=list, repr=False)

    def __post_init__(self):
        seed = self.config.random_seed
        self.rng = np.random.default_rng(seed)

    def setup(self) -> None:
        """
        Initialize the simulation.

        - Initializes all agents with the market's contract
        - Distributes initial signals from the environment
        """
        n_outcomes = self.market.contract.n_outcomes
        prior = np.ones(n_outcomes) / n_outcomes

        # Initialize agents
        for agent in self.agents:
            agent.initialise(n_outcomes, prior)

        # Distribute initial signals if configured
        if self.config.distribute_signals_once:
            self.environment.distribute_signals(self.agents)

    def step(self) -> list[Trade]:
        """
        Execute one simulation step.

        Returns:
            List of trades executed this step
        """
        step_trades = []

        # Determine which agents act this step
        acting_agents = self._select_acting_agents()

        # Agents observe and trade
        for agent in acting_agents:
            trade = agent.step(self.market)
            if trade is not None:
                step_trades.append(trade)
                self._trades.append(trade)

        # Advance market and environment time
        self.market.advance_time()
        self.environment.step()
        self._step += 1

        # Distribute new signals if not doing once at start
        if not self.config.distribute_signals_once:
            self.environment.distribute_signals(self.agents)

        return step_trades

    def _select_acting_agents(self) -> list[Agent]:
        """Select which agents act this step."""
        if self.config.agent_order == "sequential":
            # Each agent acts once per step, in order
            return self.agents

        elif self.config.agent_order == "random":
            # Shuffle order each step
            agents = self.agents.copy()
            self.rng.shuffle(agents)
            if self.config.agents_per_step is not None:
                agents = agents[:self.config.agents_per_step]
            return agents

        elif self.config.agent_order == "simultaneous":
            # All agents decide based on same prices (decisions collected then executed)
            # For now, treat same as random
            return self.agents

        else:
            return self.agents

    def run(self, progress_callback: Callable[[int, int], None] | None = None) -> SimulationResult:
        """
        Run the full simulation.

        Args:
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            SimulationResult with all collected data
        """
        self.setup()

        # Run simulation steps
        for step in range(self.config.n_steps):
            self.step()
            if progress_callback is not None:
                progress_callback(step + 1, self.config.n_steps)

        # Resolve and calculate payouts
        true_value = self.environment.resolve()
        payouts = self.market.resolve(true_value)

        # Collect results
        return self._collect_results(true_value, payouts)

    def _collect_results(self, true_value: Any, payouts: dict[str, float]) -> SimulationResult:
        """Collect all simulation data into a result object."""
        # Extract price history
        price_history = self.market.get_price_history()
        quantity_history = np.array([
            state.quantities for state in self.market.state_history
        ])

        # Agent data
        agent_final_wealth = {a.agent_id: a.wealth for a in self.agents}
        agent_pnl = {a.agent_id: payouts.get(a.agent_id, 0) for a in self.agents}
        agent_trades = {}
        for agent in self.agents:
            agent_trades[agent.agent_id] = [
                t for t in self._trades if t.agent_id == agent.agent_id
            ]

        # Theoretical aggregate (if Hayekian environment)
        theoretical_aggregate = None
        if hasattr(self.environment, 'theoretical_aggregate'):
            theoretical_aggregate = self.environment.theoretical_aggregate()

        return SimulationResult(
            config=self.config,
            n_steps=self._step,
            true_value=true_value,
            price_history=price_history,
            quantity_history=quantity_history,
            trades=self._trades,
            final_prices=self.market.get_prices(),
            agent_final_wealth=agent_final_wealth,
            agent_pnl=agent_pnl,
            agent_trades=agent_trades,
            payouts=payouts,
            total_subsidy_spent=self.market._get_subsidy_spent(),
            theoretical_aggregate=theoretical_aggregate,
            prior=np.ones(self.market.contract.n_outcomes) / self.market.contract.n_outcomes
        )

    def __repr__(self) -> str:
        return (
            f"Simulation(env={self.environment.__class__.__name__}, "
            f"n_agents={len(self.agents)}, steps={self.config.n_steps})"
        )


def run_batch(
    create_simulation: Callable[[], Simulation],
    n_runs: int,
    seeds: list[int] | None = None,
    progress_callback: Callable[[int, int], None] | None = None
) -> list[SimulationResult]:
    """
    Run multiple simulations with different random seeds.

    Args:
        create_simulation: Factory function to create a simulation
        n_runs: Number of runs
        seeds: Optional list of seeds (one per run)
        progress_callback: Optional callback(current_run, total_runs)

    Returns:
        List of SimulationResults
    """
    if seeds is None:
        seeds = list(range(n_runs))

    results = []
    for i, seed in enumerate(seeds):
        sim = create_simulation()
        sim.config.random_seed = seed
        sim.rng = np.random.default_rng(seed)
        result = sim.run()
        results.append(result)

        if progress_callback is not None:
            progress_callback(i + 1, n_runs)

    return results
