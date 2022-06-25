import os

from dotenv import load_dotenv
from jira import Issue, JIRA

from common.atlassian_utils.jira_utils.board_ci.jira_column_ci import check_rules_compliance, JiraRule

from common.atlassian_utils.tests.test_access_jira import jira_obj  # noqa

load_dotenv(f"{os.path.dirname(__file__)}/../.env")


class NoAssigneeRule(JiraRule):
    def __init__(self) -> None:
        super().__init__(rule_name="NoAssigneeRule", filter_jql='issue = "JT-3"', project_jql="PROJECT = JT")

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return iss.fields.assignee is None


class EverythingPassesRule(JiraRule):
    def __init__(self) -> None:
        super().__init__(rule_name="EverythingPassesRule", filter_jql="PROJECT = JT", project_jql="PROJECT = JT")

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        return False


def test_board_has_tickets_without_assignee(jira_obj: JIRA) -> None:
    rule = NoAssigneeRule()

    issues = jira_obj.search_issues(f"{rule.project_jql} AND {rule.filter_jql}")

    assert any(rule.does_ticket_violate_rule(iss) for iss in issues), "JT-3 should violate"


def test_board_has_tickets_without_assignee_with_check_compliance_function(jira_obj: JIRA) -> None:
    r1, r2 = NoAssigneeRule(), EverythingPassesRule()
    violation_dict = check_rules_compliance([r1, r2], jira_obj)

    assert "JT-3" in violation_dict, "JT-3 should violate"
    assert "JT-1" not in violation_dict, "JT-1 is not in the assignee filter (but is in everythingPasses)"
    assert [r1] == violation_dict["JT-3"], "JT-3 did not violate EverythingPassesRule"
