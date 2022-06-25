# type: ignore[no-any-return]
from datetime import datetime, timedelta

from jira import Issue, JIRA

from common.atlassian_utils.general import parse_atls_date_str


def field_is_none(iss: Issue, field_name: str) -> bool:
    try:
        return iss.fields.__dict__[field_name] is None
    except KeyError:
        return True  # no such field


def no_assignee(iss: Issue) -> bool:
    return field_is_none(iss, "assignee")


def no_time_estimate(iss: Issue) -> bool:
    return field_is_none(iss, "timeestimate")


def no_epic(iss: Issue) -> bool:
    return field_is_none(iss, "parent")


def no_in_progress_epic(iss: Issue) -> bool:
    if no_epic(iss):
        return False
    return iss.fields.parent.fields.status.name != "In Progress"


def no_update_for_x_days(iss: Issue, days_without_update: int) -> bool:
    updated_no_tz = parse_atls_date_str(iss.fields.updated, with_tz=False)  # in local time
    return (datetime.now() - updated_no_tz) > timedelta(days=days_without_update)


def overdue(iss: Issue) -> bool:
    if iss.fields.duedate is None:
        return False
    return parse_atls_date_str(iss.fields.duedate, with_tz=False) > datetime.now()


def flagged(iss: Issue) -> bool:
    try:
        return iss.fields.customfield_10021[0].value == "Impediment"
    except (TypeError, AttributeError):  # field is None, or structure is not as flag
        return False


def child_issues_estimate_does_not_add_up(iss: Issue) -> bool:
    raise NotImplementedError(iss)


def no_linked_page_with_prefix(iss: Issue, link_prefix: str, jira_obj: JIRA) -> bool:
    """Check if issue has linked a page with the given prefix.

    Args:
        iss: jira issue object.
        link_prefix:  Example of space link: https://jonz-test2.atlassian.net/wiki/spaces/JT/"
        jira_obj: object connecting to the jira instance

    Returns:
        True if condition holds, False otherwise.
    """
    # in the future, make sure the page is unique in the project?
    return any(rl.object.url.startswith(link_prefix) for rl in jira_obj.remote_links(iss.key))


STORY_TASK_BUG_FILTER = "type in (Bug, Story, Task)"
EPIC_FILTER = "type = Epic"
SUBTASK_FILTER = "type = Subtask"
STATUS_IN_PROGRESS_FILTER = 'status = "In Progress"'
STATUS_APPROVED_FILTER = 'status = "In Progress"'
STATUS_BLOCKED_FILTER = 'status = "Blocked"'
STATUS_TODO_FILTER = 'status = "To Do"'
STATUS_CR_FILTER = 'status = "CR"'

IN_ACTIVE_SPRINT_FILTER = "Sprint in openSprints ()"
BACKLOG_FILTER = f'{STORY_TASK_BUG_FILTER} AND Sprint is EMPTY'
