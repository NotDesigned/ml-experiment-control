"""Optional byte quotas; None means no fixed byte ceiling, not infinite disk."""


def byte_limit(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("archive byte quota must be a positive integer or null")
    return value


def exceeds(size, limit):
    return limit is not None and size > limit


def wire_limit(limit):
    # Protocol 2 clients expect an integer. Preserve their comparison logic;
    # configured quotas are separately exposed as null when no ceiling is set.
    return limit if limit is not None else 2 ** 63 - 1


def minimum_limit(*limits):
    values = [value for value in limits if value is not None]
    return min(values) if values else None
