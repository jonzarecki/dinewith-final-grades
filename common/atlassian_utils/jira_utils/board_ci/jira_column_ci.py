from dataclasses import dataclass
from typing import Dict, List

from jira import Issue, JIRA

from common.atlassian_utils.jira_utils.board_ci import rules_and_filters
from common.atlassian_utils.jira_utils.board_ci.rules_and_filters import STORY_TASK_BUG_FILTER, \
    STATUS_IN_PROGRESS_FILTER, STATUS_CR_FILTER


@dataclass(frozen=False, init=True)
class JiraRule:
    """Class for defining rules for tickets to adhere in a JIRA project."""

    rule_name: str  #: member rule name, should be unique and readable
    project_jql: str  #: member jql query for the project
    filter_jql: str  #: member jql query wanted tickets to apply the rule on

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        pass


def check_rules_compliance(rules: List[JiraRule], jira: JIRA) -> Dict[str, List[JiraRule]]:
    """Checks the compliance of the rules for the given rules.

    Args:
        rules: List of rules to check
        jira: jira object to interact with

    Returns:
        Dictionary between every violated issue-key and the list of rules it violated
    """
    violations_dict: Dict[str, List[JiraRule]] = {}  # type: ignore

    for rule in rules:
        matching_issues = jira.search_issues(f"{rule.project_jql} AND {rule.filter_jql}")
        for iss in matching_issues:
            if not rule.does_ticket_violate_rule(iss):
                continue
            violations_dict[iss.key] = violations_dict.get(iss.key, []) + [rule]

    return violations_dict


TEMP_PROJECT_JQL = "PROJECT = XXX"


class NoAssignee(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="All tickets need assignees", filter_jql=STORY_TASK_BUG_FILTER, project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_assignee(iss)


class NoEpic(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="All tickets need an epic", filter_jql=STORY_TASK_BUG_FILTER, project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_epic(iss)


class NoUpdatesForInProgress(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="No update for ticket in In Progress after 5 days, is everything OK?",
            filter_jql=f"{STORY_TASK_BUG_FILTER} AND {STATUS_IN_PROGRESS_FILTER}", project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_update_for_x_days(iss, 5)


class NoUpdatesForCR(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="No update for ticket in CR after 3 days, is everything OK?",
            filter_jql=f"{STORY_TASK_BUG_FILTER} AND {STATUS_CR_FILTER}", project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_update_for_x_days(iss, 5)


class InProgressTicketWithoutInProgressEpic(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="Found in progress ticket without in progress epic", filter_jql=STORY_TASK_BUG_FILTER,
            project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_in_progress_epic(iss)


class ChildIssuesEstimateDoesNotAddUp(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="child issues estimate does not add up", filter_jql=STORY_TASK_BUG_FILTER,
            project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.child_issues_estimate_does_not_add_up(iss)


class Overdue(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="Issue overdue!", filter_jql=STORY_TASK_BUG_FILTER,
            project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.overdue(iss)


class NoLinkedPage(JiraRule):
    def __init__(self) -> None:
        super().__init__(
            rule_name="child issues estimate does not add up",
            filter_jql=f"{STORY_TASK_BUG_FILTER} AND ({STATUS_IN_PROGRESS_FILTER} OR {STATUS_CR_FILTER})",
            project_jql=TEMP_PROJECT_JQL
        )

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return rules_and_filters.no_linked_page_with_prefix(iss, "confluence.com/", None)
