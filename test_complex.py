
from extended_numbers import C

def test_complex_equality():
    a = C( (1.0, 0) )
    assert a == C( (1, 0) )

def test_complex_nonequality():
    a = C( (0, -2.5) )
    assert a != C( (0, 0) )

def test_complex_addition():
    a = C( (1, 0) )
    b = C( (2.4, -1) )
    c = C( (3.4, -1) )
    assert a+b == c

def test_complex_subtraction():
    a = C( (3.45, -6.459) )
    b = C( (0, 1.493) )
    c = C( (3.45, -7.952) )
    assert a-b == c

def test_complex_const_mul():
    a = 2.5
    b = C( (1.2, -1) )
    c = C( (3, -2.5) )
    assert a*b == c

def test_complex_const_rmul():
    a = 2.5
    b = C( (1.2, -1) )
    c = C( (3, -2.5) )
    assert b*a == c

def test_complex_mul_real():
    a = C( (1., 0) )
    b = C( (4., 0) )
    c = C( (4., 0) )
    assert a*b == c

def test_complex_mul_imaginary():
    a = C( (0, 3.2) )
    b = C( (0, -4) )
    c = C( (12.8, 0) )
    assert a*b == c

def test_complex_mul_real_imaginary():
    a = C( (1.3, 0) )
    b = C( (0, 4.23) )
    c = C( (0, 5.499) )
    assert a*b == c

def test_complex_mul_general():
    a = C( (2.52, -48.529) )
    b = C( (-59.259, 1.2384) )
    c = C( (-89.2343664, 2878.900779) )
    assert a*b == c

