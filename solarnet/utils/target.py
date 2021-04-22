from typing import Callable, Dict, List, Union


def operator_to_lambda(operator: str) -> Callable[[float, float], bool]:
    """
    Convert an operator (in string format) to a lambda making the actual comparison.
    Support <, <=, >, >=, ==, !=.

    :param operator: An comparison operator as string
    :return: A lambda making a comparison according to given operator
    """

    comparisons_lambdas: Dict[str, Callable[[float, float], bool]] = {
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
    }

    return comparisons_lambdas[operator]


def make_classes_definitions(class_list: List[dict]) -> List[dict]:
    """
    Create a list of dict with class definitions.
    Input is a list of dict with class. Each dict as one key-value pair,
      key=class name, value=operator and flux value. E.g. {"M":">= 1e-5"}
    Output is a list of dict with class. Each dict as keys: class (class name), comparator (a lambda for comparison),
      flux (the flux to compare to).

    :param class_list: list of dicts, one class per dict
    :return: list of dicts, one class per dict
    """

    class_dict = dict(item for d in class_list for item in d.items())  # list of dict to single dict
    classes_definitions = []
    for class_name, value in class_dict.items():
        operator, flux = value.split(" ")

        classes_definitions.append({
            'class': class_name,
            'comparator': operator_to_lambda(operator),
            'flux': float(flux),
        })
    return classes_definitions


def flux_to_class_builder(class_list: List[dict], return_names: bool = False) -> Callable[[float], Union[int, str]]:
    """
    Construct a function transforming a flux in a class, according to class_list.

    :param class_list: list of dicts, one class per dict
    :param return_names: Whether to return the class name or the integer-index of the class
    :return: a function transforming a flux to a class
    """

    classes_definitions: List[dict] = make_classes_definitions(class_list)

    def flux_to_class(flux: float) -> int:
        """
        Return the correct class according to flux and classes_definitions

        :param flux: a float representing a flux
        :return: a class as int
        """

        for i, c in enumerate(classes_definitions):
            if c['comparator'](flux, c['flux']):
                return i if not return_names else c["class"]
        return -1

    return flux_to_class
