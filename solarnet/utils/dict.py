from typing import Optional


def print_dict(d: dict, title: str = None, depth: int = 0, print_type: bool = False, precision: Optional[int] = None):
    """
    Pretty print a dict, supports nested dicts
    """

    if title:
        print(f"\n{title}:")
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{'    ' * (depth + 1)}{key}:")
            print_dict(value, depth=depth + 1, print_type=print_type)
        else:
            print_value = value
            if precision is not None and isinstance(value, float):
                print_value = "{:.{prec}f}".format(value, prec=precision)
            print(f"{'    ' * (depth + 1)}{key}: {print_value}{f' ({type(value)})' if print_type else ''}")
