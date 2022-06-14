from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Participant:
    name: str
    food_grades: Dict["Participant", int] = field(default_factory=lambda: [])
    hagasha_grades: Dict["Participant", int] = field(default_factory=lambda: [])
    hospitality_grades: Dict["Participant", int] = field(default_factory=lambda: [])

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name

    def _get_norm_grades(self, grades: Dict["Participant", int]) -> Dict["Participant", float]:
        sum_grades = max(sum(grades.values()), 1)
        norm_factor = 7 * len(grades) / sum_grades
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
