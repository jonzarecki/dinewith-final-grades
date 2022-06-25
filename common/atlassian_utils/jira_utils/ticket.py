from jira import Issue


def get_ticket_url_in_board(iss: Issue) -> str:
    """Returns the tickets' url as 'selected' in the board."""
    return f"{iss.self.split('rest')[0]}/jira/software/projects/{iss.fields.project}/boards/" \
           f"{iss.fields.customfield_10020[0].boardId}?selectedIssue={iss.key}"
