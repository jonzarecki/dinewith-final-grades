import logging
from datetime import datetime, timedelta
from typing import List, Optional, Callable

from jira import Comment, Issue, JIRA

from common.atlassian_utils.general import parse_atls_date_str
from common.atlassian_utils.jira_utils.board_ci import rules_and_filters
from common.atlassian_utils.jira_utils.board_ci.rules_and_filters import IN_ACTIVE_SPRINT_FILTER, STORY_TASK_BUG_FILTER


def flag_issue(iss_key: str, jira_obj: JIRA) -> None:
    iss = jira_obj.issue(iss_key)
    if not rules_and_filters.flagged(iss):  # type: ignore
        try:
            iss.fields.customfield_10021 = [{"value": "Impediment"}]
            iss.update(jira=jira_obj)
        except:  # noqa
            print("Failed to flag")


def write_jira_comment(iss_key: str, comment_body: str, jira_obj: JIRA) -> None:
    jira_obj.add_comment(iss_key, comment_body)


def check_if_comment_already_exists(iss: Issue, violation_body: str) -> Optional[str]:
    comments: List[Comment] = iss.fields.comment.comments  # type: ignore
    now = datetime.now()

    for c in comments:
        if c.body == violation_body:
            creation_date_no_tz = parse_atls_date_str(c.created, with_tz=False)  # in local time
            if now - creation_date_no_tz < timedelta(days=2):
                return str(c.id)
    return None


def transition_parent_tickets_to_match_children_in_open_sprint(status_order: List[str], project_key: str,
                                                               jira_obj: JIRA,
                                                               callback_f: Optional[Callable] = None) -> None:
    """Transition parent tickets to match children in open sprint."""
    matching_issues = jira_obj.search_issues(f"project={project_key} AND {IN_ACTIVE_SPRINT_FILTER} AND "
                                             f"{STORY_TASK_BUG_FILTER}")

    if len(matching_issues) == 0:
        return

    transitions = [(t['id'], t['name']) for t in jira_obj.transitions(matching_issues[0])]
    trans_ids = [t[0] for t in transitions]
    trans_names = [t[1] for t in transitions]
    assert all([status in trans_names for status in status_order]), "all statuses should exist in the transitions"

    for iss in matching_issues:
        try:
            subtasks = iss.fields.subtasks
            if len(subtasks) == 0:
                continue
        except AttributeError:
            continue  # no subtasks

        try:
            logging.info(f"iss - {iss}")
            min_status_idx = min(status_order.index(sub_t.fields.status.name) for sub_t in iss.fields.subtasks)
            min_status_name = status_order[min_status_idx]  # children minimum status

            if iss.fields.status.name == min_status_name:
                continue  # don't do anything

            jira_obj.transition_issue(iss, trans_ids[trans_names.index(min_status_name)])
            if callback_f is not None:
                callback_f(iss, min_status_name)
        except ValueError:
            raise AssertionError(f"{iss} has a subtask with a status not matching any in \n {status_order}")


def write_violation_as_comment(iss: Issue, violation_body: str, jira_obj: JIRA) -> bool:
    """Writes the violation as a UNIQUE comment and flags the issue.

    Check if violation exists, if it does and was written more than 2 days ago, delete it and write a new one.
    """
    # check if violation exists
    violation_comment_id = check_if_comment_already_exists(iss, violation_body)
    if violation_comment_id is not None:
        return False

    write_jira_comment(iss.key, violation_body, jira_obj)
    flag_issue(iss.key, jira_obj)
    return True
