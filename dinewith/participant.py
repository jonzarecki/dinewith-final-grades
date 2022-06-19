from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Participant:
    name: str
    is_grader: bool  # not used atm, for clarity in objects
    is_gradee: bool  # not used atm, for clarity in objects
    food_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # pylint
    hagasha_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # noqa
    hospitality_grades: Dict["Participant", int] = field(default_factory=lambda: dict())  # noqa

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def _get_norm_grades(self, grades: Dict["Participant", int]) -> Dict["Participant", float]:
        rel_grades = {p: g for (p, g) in grades.items() if g != 0}  # relevant is non-zero
        sum_grades = max(1.0, sum(rel_grades.values()))
        norm_factor = float(7 * len(rel_grades)) / sum_grades

        return {part: grade * norm_factor for (part, grade) in grades.items()}

    def norm_food_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.food_grades)

    def norm_hagasha_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.hagasha_grades)

    def norm_hospitality_grades(self) -> Dict["Participant", float]:
        return self._get_norm_grades(self.hospitality_grades)
