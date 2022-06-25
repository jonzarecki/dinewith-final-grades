import os

from dotenv import load_dotenv
from jira import JIRA

from common.atlassian_utils.jira_utils.on_ticket_communication import ticket_has_unanswered_question

from common.atlassian_utils.tests.test_access_jira import jira_obj  # noqa

load_dotenv(f"{os.path.dirname(__file__)}/../.env")


def test_ticket_has_unanswered_questions(jira_obj: JIRA) -> None:
    iss = jira_obj.issue("JT-3")
    assert ticket_has_unanswered_question(jira_obj, iss)
