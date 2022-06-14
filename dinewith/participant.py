from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Participant:
    name: str
    food_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # pylint
    hagasha_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # noqa
    hospitality_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # noqa

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def _get_norm_grades(self, grades: Dict["Participant", int]) -> Dict["Participant", float]:
        sum_grades = max(1.0, sum(grades.values()))
        norm_factor = float(7 * len(grades)) / sum_grades if all(v != 0 for v in grades.values()) else 1.0

        return {part: grade * norm_factor for (part, grade) in grades.items()}

    @property
    def norm_food_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.food_grades)

    @property
    def norm_hagasha_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.hagasha_grades)

    @property
    def norm_hospitality_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.hospitality_grades)
