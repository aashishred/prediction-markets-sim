"""
Contract type definitions for prediction markets.

Supports various contract structures as described in the theoretical framework:
- Binary: Pays 1 if event occurs, 0 otherwise
- Multinomial: Mutually exclusive outcomes, prices sum to 1
- Continuous: Pays based on final value of a continuous variable
- Range: Discretises a continuous outcome into bands
- Conditional: Only resolves if a condition is met
- Combinatorial: Complex logical combinations (future extension)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ContractType(Enum):
    """Enumeration of supported contract types."""
    BINARY = auto()
    MULTINOMIAL = auto()
    CONTINUOUS = auto()
    RANGE = auto()
    CONDITIONAL = auto()
    COMBINATORIAL = auto()


class ResolutionStatus(Enum):
    """Status of contract resolution."""
    PENDING = auto()      # Not yet resolved
    RESOLVED = auto()     # Resolved with outcome
    VOIDED = auto()       # Voided (e.g., condition not met)


@dataclass
class Contract(ABC):
    """
    Abstract base class for all contract types.

    A contract defines what is being predicted and how payouts are determined.
    """
    name: str
    description: str = ""
    resolution_status: ResolutionStatus = field(default=ResolutionStatus.PENDING)
    resolved_outcome: Any = None

    @property
    @abstractmethod
    def contract_type(self) -> ContractType:
        """Return the type of this contract."""
        pass

    @property
    @abstractmethod
    def n_outcomes(self) -> int:
        """Return the number of possible outcomes."""
        pass

    @property
    @abstractmethod
    def outcome_names(self) -> list[str]:
        """Return human-readable names for each outcome."""
        pass

    @abstractmethod
    def payout(self, outcome_index: int) -> float:
        """
        Calculate the payout for holding one share of the given outcome.

        Args:
            outcome_index: Index of the outcome to check

        Returns:
            Payout amount (typically 0 or 1 for binary/multinomial)
        """
        pass

    @abstractmethod
    def resolve(self, outcome: Any) -> None:
        """
        Resolve the contract with the given outcome.

        Args:
            outcome: The realised outcome (interpretation depends on contract type)
        """
        pass

    def is_resolved(self) -> bool:
        """Check if the contract has been resolved."""
        return self.resolution_status != ResolutionStatus.PENDING


@dataclass
class BinaryContract(Contract):
    """
    A binary contract that pays 1 if an event occurs, 0 otherwise.

    This is the simplest and most common prediction market contract.
    Example: "Will candidate X win the election?"
    """
    event_description: str = ""

    @property
    def contract_type(self) -> ContractType:
        return ContractType.BINARY

    @property
    def n_outcomes(self) -> int:
        return 2

    @property
    def outcome_names(self) -> list[str]:
        return ["No", "Yes"]

    def payout(self, outcome_index: int) -> float:
        if not self.is_resolved():
            raise ValueError("Contract not yet resolved")
        if outcome_index not in (0, 1):
            raise ValueError(f"Invalid outcome index {outcome_index} for binary contract")
        # outcome_index 1 = Yes, 0 = No
        return 1.0 if outcome_index == self.resolved_outcome else 0.0

    def resolve(self, outcome: bool | int) -> None:
        """
        Resolve the binary contract.

        Args:
            outcome: True/1 if event occurred, False/0 otherwise
        """
        self.resolved_outcome = int(bool(outcome))
        self.resolution_status = ResolutionStatus.RESOLVED


@dataclass
class MultinomialContract(Contract):
    """
    A multinomial contract with mutually exclusive outcomes.

    Exactly one outcome will occur; prices should sum to 1.
    Example: "Which candidate will win? {A, B, C, D}"
    """
    outcomes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.outcomes) < 2:
            raise ValueError("Multinomial contract requires at least 2 outcomes")

    @property
    def contract_type(self) -> ContractType:
        return ContractType.MULTINOMIAL

    @property
    def n_outcomes(self) -> int:
        return len(self.outcomes)

    @property
    def outcome_names(self) -> list[str]:
        return self.outcomes.copy()

    def payout(self, outcome_index: int) -> float:
        if not self.is_resolved():
            raise ValueError("Contract not yet resolved")
        if outcome_index < 0 or outcome_index >= self.n_outcomes:
            raise ValueError(f"Invalid outcome index {outcome_index}")
        return 1.0 if outcome_index == self.resolved_outcome else 0.0

    def resolve(self, outcome: int | str) -> None:
        """
        Resolve the multinomial contract.

        Args:
            outcome: Either the index or name of the winning outcome
        """
        if isinstance(outcome, str):
            if outcome not in self.outcomes:
                raise ValueError(f"Unknown outcome: {outcome}")
            outcome = self.outcomes.index(outcome)
        if outcome < 0 or outcome >= self.n_outcomes:
            raise ValueError(f"Invalid outcome index: {outcome}")
        self.resolved_outcome = outcome
        self.resolution_status = ResolutionStatus.RESOLVED


@dataclass
class ContinuousContract(Contract):
    """
    A continuous contract that pays based on the final value of a variable.

    The payout is the actual value (possibly scaled/bounded).
    Example: "What will the inflation rate be?"
    """
    min_value: float = 0.0
    max_value: float = 1.0

    def __post_init__(self):
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")

    @property
    def contract_type(self) -> ContractType:
        return ContractType.CONTINUOUS

    @property
    def n_outcomes(self) -> int:
        # Continuous contracts are represented as a single "outcome"
        # whose value is the continuous variable
        return 1

    @property
    def outcome_names(self) -> list[str]:
        return [f"Value in [{self.min_value}, {self.max_value}]"]

    def payout(self, outcome_index: int = 0) -> float:
        """
        For continuous contracts, payout is the normalised resolved value.

        Returns value in [0, 1] where 0 = min_value, 1 = max_value.
        """
        if not self.is_resolved():
            raise ValueError("Contract not yet resolved")
        if outcome_index != 0:
            raise ValueError("Continuous contracts have only one 'outcome'")
        # Normalise to [0, 1]
        return (self.resolved_outcome - self.min_value) / (self.max_value - self.min_value)

    def resolve(self, outcome: float) -> None:
        """
        Resolve the continuous contract.

        Args:
            outcome: The final value (will be clamped to [min_value, max_value])
        """
        self.resolved_outcome = max(self.min_value, min(self.max_value, outcome))
        self.resolution_status = ResolutionStatus.RESOLVED


@dataclass
class RangeContract(Contract):
    """
    A range contract that discretises a continuous outcome into bands.

    Converts a continuous variable into a multinomial by defining buckets.
    Example: "Will inflation be: <2%, 2-3%, 3-4%, >4%?"
    """
    boundaries: list[float] = field(default_factory=list)
    labels: list[str] | None = None

    def __post_init__(self):
        if len(self.boundaries) < 1:
            raise ValueError("Range contract requires at least 1 boundary")
        # Boundaries should be sorted
        self.boundaries = sorted(self.boundaries)
        # Generate default labels if not provided
        if self.labels is None:
            self.labels = []
            self.labels.append(f"< {self.boundaries[0]}")
            for i in range(len(self.boundaries) - 1):
                self.labels.append(f"{self.boundaries[i]} - {self.boundaries[i+1]}")
            self.labels.append(f">= {self.boundaries[-1]}")
        if len(self.labels) != len(self.boundaries) + 1:
            raise ValueError("Number of labels must be number of boundaries + 1")

    @property
    def contract_type(self) -> ContractType:
        return ContractType.RANGE

    @property
    def n_outcomes(self) -> int:
        return len(self.boundaries) + 1

    @property
    def outcome_names(self) -> list[str]:
        return self.labels.copy()

    def _value_to_bucket(self, value: float) -> int:
        """Convert a continuous value to its bucket index."""
        for i, boundary in enumerate(self.boundaries):
            if value < boundary:
                return i
        return len(self.boundaries)

    def payout(self, outcome_index: int) -> float:
        if not self.is_resolved():
            raise ValueError("Contract not yet resolved")
        if outcome_index < 0 or outcome_index >= self.n_outcomes:
            raise ValueError(f"Invalid outcome index {outcome_index}")
        winning_bucket = self._value_to_bucket(self.resolved_outcome)
        return 1.0 if outcome_index == winning_bucket else 0.0

    def resolve(self, outcome: float) -> None:
        """
        Resolve the range contract with a continuous value.

        Args:
            outcome: The continuous value that determines which bucket wins
        """
        self.resolved_outcome = outcome
        self.resolution_status = ResolutionStatus.RESOLVED


@dataclass
class ConditionalContract(Contract):
    """
    A conditional contract that only resolves if a condition is met.

    "If P is implemented, will Y occur?"
    If P is not implemented, the contract is voided.
    """
    condition_description: str = ""
    consequence_description: str = ""
    condition_met: bool | None = None

    @property
    def contract_type(self) -> ContractType:
        return ContractType.CONDITIONAL

    @property
    def n_outcomes(self) -> int:
        # Yes/No for the consequence, plus implicit "voided" state
        return 2

    @property
    def outcome_names(self) -> list[str]:
        return ["No (given condition)", "Yes (given condition)"]

    def payout(self, outcome_index: int) -> float:
        if not self.is_resolved():
            raise ValueError("Contract not yet resolved")
        if self.resolution_status == ResolutionStatus.VOIDED:
            # If voided, all positions are unwound at purchase price (return 0 profit)
            raise ValueError("Contract voided - no payout")
        if outcome_index not in (0, 1):
            raise ValueError(f"Invalid outcome index {outcome_index}")
        return 1.0 if outcome_index == self.resolved_outcome else 0.0

    def resolve(self, condition_met: bool, consequence: bool | None = None) -> None:
        """
        Resolve the conditional contract.

        Args:
            condition_met: Whether the condition (P) was met
            consequence: If condition met, whether the consequence (Y) occurred
        """
        self.condition_met = condition_met
        if not condition_met:
            self.resolution_status = ResolutionStatus.VOIDED
            self.resolved_outcome = None
        else:
            if consequence is None:
                raise ValueError("Must provide consequence when condition is met")
            self.resolved_outcome = int(bool(consequence))
            self.resolution_status = ResolutionStatus.RESOLVED
