from fraction import Fraction

def test_eq():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/3")
    assert a == b

def test_eq2():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/2")
    assert a != b

def test_reduce():
    a = Fraction.from_string("2/6")
    b = Fraction.from_string("1/3")
    assert a == b

def test_reduce2():
    a = Fraction.from_string("2/6")
    b = Fraction.from_string("-1/-3")
    assert a == b

def test_addition():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/5")
    c = Fraction.from_string("8/15")
    assert a + b == c

def test_addition2():
    a = Fraction.from_string("3/10")
    b = Fraction.from_string("1/6")
    c = Fraction.from_string("7/15")
    assert a + b == c

def test_subtraction():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/5")
    c = Fraction.from_string("2/15")
    assert a - b == c

def test_subtraction2():
    a = Fraction.from_string("3/10")
    b = Fraction.from_string("1/6")
    c = Fraction.from_string("2/15")
    assert a - b == c

def test_multiplication():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/5")
    c = Fraction.from_string("1/15")
    assert a * b == c

def test_multiplication2():
    a = Fraction.from_string("3/10")
    b = Fraction.from_string("1/6")
    c = Fraction.from_string("1/20")
    assert a * b == c

def test_division():
    a = Fraction.from_string("1/3")
    b = Fraction.from_string("1/5")
    c = Fraction.from_string("5/3")
    assert a / b == c

def test_division2():
    a = Fraction.from_string("3/10")
    b = Fraction.from_string("1/6")
    c = Fraction.from_string("9/5")
    assert a / b == c

