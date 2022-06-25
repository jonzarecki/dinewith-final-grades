import os
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from jira import Issue, JIRA

from common.atlassian_utils.jira_utils.ticket import get_ticket_url_in_board
from common.atlassian_utils.jira_utils.user import user_exists

load_dotenv(f"{os.path.dirname(__file__)}/../.env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}


def build_issue_into_mail(iss: Issue) -> str:
    title = f"[{iss.fields.summary}]({get_ticket_url_in_board(iss)}): done by - " \
            f"{iss.fields.assignee if iss.fields.assignee else 'Unassigned'}"
    body = f"\n     {iss.fields.description}" if iss.fields.description else ""

    return title + body


def build_daily_email(jira: JIRA, project_key: str, team_members: List[str]) -> str:
    """Builds a string with a 'daily email' with what each member is up to."""
    if not project_exists(jira, project_key):
        return f"**No such project {project_key}**"

    message = ""
    message += f"**{datetime.now().strftime('%m/%d/%Y')}'s Daily** \n"

    for member in team_members:
        if user_exists(jira, member):
            issues = jira.search_issues(f"project={project_key} and assignee={member} and Sprint in openSprints ()")
            member_str = "\n".join([build_issue_into_mail(iss) for iss in issues])
        else:
            member_str = f"User {member} doesn't exist"
        message += f"\n\n {member_str}"

    return message


def project_exists(jira: JIRA, project_key: str) -> bool:
    try:
        jira.project(project_key)
        return True
    except jira.exceptions.JIRAError:
        return False


def main() -> None:
    jira_obj = JIRA(
        server=os.environ["JIRA_URL"], basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_PASSWORD"])
    )
    _msg = build_daily_email(jira_obj, "JT", ["557058:8f50afc7-9921-4c17-8d6a-6cce70d675fd"])
    jira = JIRA(server=os.environ["JIRA_URL"], basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_PASSWORD"]))
    iss = jira.issue("JT-3")  # noqa
    s = jira.search_issues("project=JT and assignee != currentUser()")  # noqa
    a = 1  # noqa


if __name__ == "__main__":
    main()
