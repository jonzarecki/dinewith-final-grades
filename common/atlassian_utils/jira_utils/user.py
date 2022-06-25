from jira import JIRA, User
from jira.resources import Resource


class JiraUser(User):
    """A Jira user, working with atlassian cloud."""

    def __init__(self, options, session, raw=None):  # type: ignore
        Resource.__init__(self, "user?accountId={0}", options, session)  # noqa
        if raw:
            self._parse_raw(raw)


def load_user(jira: JIRA, account_id: str) -> User:
    user = JiraUser(jira._options, jira._session)  # noqa
    user.find(account_id, params={})
    return user


def user_exists(jira: JIRA, account_id: str) -> bool:
    try:
        load_user(jira, account_id)
        return True
    except jira.exceptions.JIRAError:
        return False
