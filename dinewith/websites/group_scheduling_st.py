import datetime
import os
import sys
from typing import List

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dinewith.participant import Participant  # pylint: disable=wrong-import-position

# TODO: Not sure we need this. A better google sheets + connection to sheets might be good enough

st.title("Group Scheduling")

singles_group = st.checkbox("קבוצת יחידים?", value=True)

st.text(
    """
אם אתם בסקר הזה כנראה אתם מעוניינים לארח ולהתארח בתחרות בישול ביתית מול הטובים שבחברים-של-חברים שלכם!

העונה הקרובה (3) יהיו כמה פורמטים לעונה, ונבחר איזה לפי הביקוש:
1. עונת יחידים+ - מתארחים לבד, ניתן להביא חבר שיעזור לבשל ויתארח בארוחה שבישל (ככה שבכל ארוחה יש +1 של המארח).
1. עונת זוגות - זוג (גם זוגות חברים הולך) מארח ומתארח בכל הארוחות.
3. עונת זוגות LGBT+ - כמו זוגות, רק יותר טוב.

כמה חידודים:
1. בכל קבוצה יהיו 3-4 זוגות/יחידים שיתארחו/יארחו ל3-4 ארוחות לאורך 3 שבועות
2. חברי קבוצה לא אמורים להכיר אחד את השני, אם הגעתם דרך חברים שנרשמו - עדכנו אותנו!
3. ארוחות בשעות הערב, סביבות 20-21
4. "ארוחת ערב" - הגדרה:‏ ראשונה, עיקרית+סייד-דיש, קינוח, אלכוהול. יש תמונות ותפריטים לדוגמה למטה :)

הפורמט נמצא כאן:
https://docs.google.com/document/d/1rujfaJl_bDaEQjux-3Xz6Dolxun71QjRDv74o5IcaWQ/edit?usp=sharing


אם מעניין אתכם להשתתף ואתם רציניים נתחיל בלמלא את האימייל שלכם :)
"""
)

email = st.text_input("Enter Email:")

participants_num = int(st.number_input("מה מספר המשתתפים?", value=4))
if not singles_group and participants_num % 2 != 0:
    st.error("מס' המשתתפים בקבוצות זוגיות צריך להיות זוגי!")
    st.stop()

graders: List[Participant] = []
gradees: List[Participant] = []

# calculate grades using $graders and $gradees


df = pd.DataFrame(
    {
        "date": pd.date_range(datetime.datetime(2022, 6, 22), datetime.date.today()),
        "le-areh": [True for _ in range(3)],
        "lehit-areh": [False for _ in range(3)],
    }
)
df["le-areh"] = df["le-areh"].astype(bool)
my_list_of_options = ["Option 1", "Option 2", "Option 3"]
gb = GridOptionsBuilder.from_dataframe(df)
dropdownlst = ("a", "b")
sel_mode = st.radio("Selectio Type", options=["single", "multiple"])
gb.configure_column("le-areh", editable=True, selection_mode="multiple", use_checkbox=True)

# gb.configure_column('le-areh', checkboxSelection=True)
# gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=[1, 2, 3])
# gb.configure_default_column(selection_mode="multiple", checkboxSelection=True, headerCheckboxSelection=False)


grid_return = AgGrid(
    df,
    gridOptions=gb.build(),
    editable=True,
)

# gb = GridOptionsBuilder.from_dataframe(df)
# # gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
# gb.configure_selection('single')
# gb = GridOptionsBuilder.from_dataframe(df)
# gb.configure_selection('single')
# AgGrid(
#     df,
#     gridOptions=gb.build(),
#     # this override the default VALUE_CHANGED
#     update_mode=GridUpdateMode.MODEL_CHANGED
# )
# # gridOptions = gb.build()
#
# # # set up data return/update modes
# # return_mode_value = DataReturnMode.__members__['FILTERED_AND_SORTED']
# # update_mode_value = GridUpdateMode.__members__['FILTERING_CHANGED']
# #
# # data_return_mode = return_mode_value
# #
# # grid_data = AgGrid(dates_df,
# #                    gridOptions=gridOptions,
# #                    width='100%',
# #                    data_return_mode=return_mode_value,
# #                    update_mode=update_mode_value)
# #
# # # grid_return = pd.DataFrame(grid_data['data'])
# #
# # part_grades_new = grid_return["data"]
