# pylint: disable=redefined-outer-name
import os

import pytest
from atlassian import Jira
from dotenv import load_dotenv
from jira import JIRA

load_dotenv(f"{os.path.dirname(__file__)}/../.env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}


@pytest.fixture()
def jira_obj() -> JIRA:
    return JIRA(server=os.environ["JIRA_URL"], basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_PASSWORD"]))


def test_atlassian_api_jira() -> None:
    print(os.environ)
    jira = Jira(
        url=os.environ["JIRA_URL"],
        username=os.environ["JIRA_USERNAME"],
        password=os.environ["JIRA_PASSWORD"],
        cloud=True,
    )
    data1 = jira.get_all_projects(None, expand=False)
    print(data1)
    jql_request = "PROJECT = JT AND status NOT IN (Closed, Resolved) ORDER BY issuekey"
    data = jira.jql(jql_request)
    print(data)


def test_python_jira(jira_obj: JIRA) -> None:
    projects = jira_obj.projects()
    assert len(projects) > 0
    found_issues = jira_obj.search_issues("PROJECT = JT AND status NOT IN (Closed, Resolved) ORDER BY issuekey")
    assert len(found_issues) > 1
