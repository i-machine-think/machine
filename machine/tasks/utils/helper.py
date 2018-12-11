# Helper Functions
def flatten(l):
    """Flattens a list of element or lists into a list of elements."""
    out = []
    for e in l:
        if not isinstance(e, list):
            e = [e]
        out.extend(e)
    return out


def repeat(s, n, start=1):
    """Repeats a string multiple times by adding a iter index to the name."""
    return ["{}_{}".format(s, i) for i in range(start, n + start)]


def filter_dict(d, remove):
    """Filters our the key of a dictionary."""
    return {k: v for k, v in d.items() if k not in remove}
