from datetime import datetime


def parse_atls_date_str(date_str: str, with_tz: bool = False) -> datetime:
    """Parse date string in atlassian's format to a datetime object."""
    creation_date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone()
    if with_tz:
        return creation_date
    else:
        return creation_date.replace(tzinfo=None)  # in local time, without tz object
