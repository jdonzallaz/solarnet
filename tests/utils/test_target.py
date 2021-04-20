from solarnet.utils.target import operator_to_lambda


# TODO: test other target functions


def test_operator_to_lambda():
    fn = operator_to_lambda("<")
    assert fn(1, 2)
    assert not fn(1, 1)
    assert not fn(2, 1)

    fn = operator_to_lambda("<=")
    assert fn(1, 2)
    assert fn(1, 1)
    assert not fn(2, 1)

    fn = operator_to_lambda(">")
    assert not fn(1, 2)
    assert not fn(1, 1)
    assert fn(2, 1)

    fn = operator_to_lambda(">=")
    assert not fn(1, 2)
    assert fn(1, 1)
    assert fn(2, 1)

    fn = operator_to_lambda("==")
    assert not fn(1, 2)
    assert fn(1, 1)
    assert not fn(2, 1)

    fn = operator_to_lambda("!=")
    assert fn(1, 2)
    assert not fn(1, 1)
    assert fn(2, 1)
