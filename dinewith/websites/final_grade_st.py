import itertools
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dinewith.participant import Participant  # pylint: disable=wrong-import-position

st.title("Final Grade Computation")

singles_group = st.checkbox("קבוצת יחידים?", value=True)

participants_num = int(st.number_input("מה מספר המשתתפים?", value=4))
if not singles_group and participants_num % 2 != 0:
    st.error("מס' המשתתפים בקבוצות זוגיות צריך להיות זוגי!")
    st.stop()

graders: List[Participant] = []
gradees: List[Participant] = []
if singles_group:
    _participants: List[Participant] = []
    for i in range(participants_num):
        i_part_name = st.text_input(f"שם משתתף ה-{i}", value=i)
        _participants.append(Participant(i_part_name, is_grader=True, is_gradee=True))  # type: ignore

    graders = _participants
    gradees = _participants

    pname_without_dups = [
        (p_cpl[0].name, p_cpl[1].name)
        for p_cpl in itertools.product(graders, gradees)
        if p_cpl[0].name != p_cpl[1].name
    ]
else:  # couples group
    couples_num = int(participants_num / 2)

    cpls: List[Tuple[Participant, Participant]] = []
    for cpl_i in range(couples_num):
        for i in range(1, 3):
            i_part_name = st.text_input(f"שם משתתף ה-{i} בזוג ה-{cpl_i}", value=2 * cpl_i + i)
            graders.append(Participant(i_part_name, is_grader=True, is_gradee=False))  # type: ignore
        cpls.append((graders[-2], graders[-1]))
        gradees.append(Participant(f"{graders[-2].name} - {graders[-1].name}", is_grader=False, is_gradee=True))

    pname_without_dups = []
    for cpl1, cpl2 in itertools.product(cpls, cpls):
        if cpl1[0].name == cpl2[0].name and cpl1[1].name == cpl2[1].name:
            print(cpl1, cpl2)
            continue
        pname_without_dups.append((cpl2[0].name, f"{cpl1[0].name} - {cpl1[1].name}"))
        pname_without_dups.append((cpl2[1].name, f"{cpl1[0].name} - {cpl1[1].name}"))
    print(pname_without_dups)

# calculate grades using $graders and $gradees

part_grades = pd.DataFrame(index=pname_without_dups)
part_grades["grader"] = [cpl_name[0] for cpl_name in pname_without_dups]
part_grades["gradee"] = [cpl_name[1] for cpl_name in pname_without_dups]
part_grades["food"] = 0
part_grades["hagasha"] = 0
part_grades["hospitality"] = 0
part_grades = part_grades.sort_values(by=["grader"])

grid_return = AgGrid(part_grades, editable=True)
part_grades_new = grid_return["data"]

for p in graders:
    p_grades = part_grades_new[part_grades_new["grader"] == p.name]

    def build_grades_dict(col_name: str) -> Dict[Participant, int]:
        return {
            p_o: p_grades[p_grades["gradee"] == p_o.name][col_name].iloc[0]
            for p_o in gradees
            if len(p_grades[p_grades["gradee"] == p_o.name]) != 0
        }

    p.food_grades = build_grades_dict("food")
    p.hagasha_grades = build_grades_dict("hagasha")
    p.hospitality_grades = build_grades_dict("hospitality")


def display_grades(norm_grades_f, grades_attr_name) -> None:  # type: ignore
    """Displays normalized and non-normalized grades for a category."""
    st.subheader(f"{grades_attr_name} grades")
    grades = pd.DataFrame(
        {
            p.name: [
                p.__dict__[f"{grades_attr_name}_grades"][p_o]
                if (p_o in p.__dict__[f"{grades_attr_name}_grades"])
                else 0.0
                for p_o in gradees
            ]
            for p in graders
        },
        index=[p.name for p in gradees],
    )
    grades["total_score"] = grades.sum(axis=1)
    st.table(grades)
    st.text("Normalized:")
    norm_grades = pd.DataFrame(
        {p.name: [norm_grades_f(p)[p_o] if (p_o in norm_grades_f(p)) else 0.0 for p_o in gradees] for p in graders},
        index=[p.name for p in gradees],
    )
    norm_grades["total_score"] = norm_grades.sum(axis=1)

    st.table(norm_grades)


display_grades(Participant.norm_food_grades, "food")
display_grades(Participant.norm_hagasha_grades, "hagasha")
display_grades(Participant.norm_hospitality_grades, "hospitality")

st.subheader("Reported grades - for review")
st.table(part_grades_new)
