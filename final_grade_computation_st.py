import itertools
from typing import Dict, List

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

from dinewith.participant import Participant

st.title("Final Grade Computation")

singles_group = st.checkbox("קבוצת יחידים?")

# assume it's singles

participants_num = int(st.number_input("מה מספר המשתתפים?", value=0))

participants: List[Participant] = []
for i in range(participants_num):
    i_part_name = st.text_input(f"שם משתתף ה-{i}", value=i)
    participants.append(Participant(i_part_name))  # type: ignore

pname_without_dups = [
    f"{p_cpl[0].name} - {p_cpl[1].name}"
    for p_cpl in itertools.product(participants, participants)
    if p_cpl[0].name != p_cpl[1].name
]
part_grades = pd.DataFrame(index=pname_without_dups)
part_grades["grader"] = [cpl_name.split(" - ")[0] for cpl_name in pname_without_dups]
part_grades["gradee"] = [cpl_name.split(" - ")[1] for cpl_name in pname_without_dups]
part_grades["food"] = 0
part_grades["hagasha"] = 0
part_grades["hospitality"] = 0
print(part_grades)

grid_return = AgGrid(part_grades, editable=True)
part_grades_new = grid_return["data"]

for p in participants:
    p_grades = part_grades_new[part_grades_new["grader"] == p.name]

    def build_grades_dict(col_name: str) -> Dict[Participant, int]:
        return {p_o: p_grades[p_grades["gradee"] == p_o.name][col_name].iloc[0] for p_o in participants if p != p_o}

    p.food_grades = build_grades_dict("food")
    p.hagasha_grades = build_grades_dict("hagasha")
    p.hospitality_grades = build_grades_dict("hospitality")


def display_grades(norm_grades_f, grades_attr_name) -> None:  # type: ignore
    """Displays normalized and non-normalized grades for a category."""
    st.subheader(f"{grades_attr_name} grades")
    grades = pd.DataFrame(
        {
            p.name: [p.__dict__[f"{grades_attr_name}_grades"][p_o] if p_o != p else 0.0 for p_o in participants]
            for p in participants
        },
        index=[p.name for p in participants],
    )
    grades["total_score"] = grades.sum(axis=1)
    st.table(grades)
    st.text("Normalized:")
    norm_grades = pd.DataFrame(
        {p.name: [norm_grades_f(p)[p_o] if p_o != p else 0.0 for p_o in participants] for p in participants},
        index=[p.name for p in participants],
    )
    norm_grades["total_score"] = norm_grades.sum(axis=1)

    st.table(norm_grades)


display_grades(Participant.norm_food_grades, "food")
display_grades(Participant.norm_hagasha_grades, "hagasha")
display_grades(Participant.norm_hospitality_grades, "hospitality")

st.subheader("Reported grades - for review")
st.table(part_grades_new)
