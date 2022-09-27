import pytest

from polynomial import Polynomial
from polynomial import RationalPolynomial

def test_rat_polynomial_definition():
    a = Polynomial.from_string("7 + x^3")
    b = Polynomial.from_string("-x^2 - 8*x")
    c = RationalPolynomial.from_string("(-7 - x^3)/(x^2 + 8*x)")
    assert a/b == c
	
def test_rat_polynomial_definition2():
    a = Polynomial.from_string("-7*x")
    b = Polynomial.from_string("-x^3 - 8*x")
    c = RationalPolynomial.from_string("(7)/(x^2 + 8)")
    assert a/b == c
	
def test_rat_polynomial_definition2():
    a = Polynomial.from_string("-49*x^3+7*x")
    b = Polynomial.from_string("-7*x^2+1")
    c = RationalPolynomial.from_string("(7*x)/(1)")
    assert a/b == c
	
def test_rat_polynomial_eq():
    a = RationalPolynomial.from_string("(-4 + x^2)/(5 - x)")
    b = RationalPolynomial.from_string("(x^2 - 4)/(-x + 5)")
    assert a == b
	
def test_rat_polynomial_eq2():
    a = RationalPolynomial.from_string("(-4 + x^2)/(5 - x)")
    b = RationalPolynomial.from_string("(-4 - x^2)/(5 - x)")
    assert a != b
	
def test_rat_polynomial_eq3():
    a = RationalPolynomial.from_string("(-4 + x^2)/(5 - x)")
    b = RationalPolynomial.from_string("(4 - x^2)/(x - 5)")
    assert a == b
	
def test_rat_polynomial_addition():
    a = RationalPolynomial.from_string("(5 - 6*x)/(1 + x)")
    b = RationalPolynomial.from_string("(-4 + 3*x^2)/(1 - x)")
    c = RationalPolynomial.from_string("(1 - 15*x + 9*x^2 + 3*x^3)/(1 - x^2)")
    assert a + b != c
	
def test_rat_polynomial_addition2():
    a = RationalPolynomial.from_string("(-x^3 + 8*x + 3)/(3 - x)")
    b = RationalPolynomial.from_string("(-6*x^3 - 25*x^2 - 2*x - 9)/(7*x - 1)")
    c = RationalPolynomial.from_string("(x^3 - 5*x^2 + 2*x - 10)/(7*x - 1)")
    assert a + b == c
	
def test_rat_polynomial_addition3():
    a = RationalPolynomial.from_string("(-3 - x^2 + 2*x^3)/(19*x^5)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(19*x^5)")
    c = RationalPolynomial.from_string("(-2 + 6*x^2 + 2*x^3)/(19*x^5)")
    assert a + b == c
	
def test_rat_polynomial_addition4():
    a = RationalPolynomial.from_string("(0+0*x^3)/(-3 -x -7*x^2)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(9 + 3*x + 17*x^2)")
    c = RationalPolynomial.from_string("(1 + 7*x^2)/(9 + 3*x + 17*x^2)")
    assert a + b == c
	
def test_rat_polynomial_subtraction():
    a = RationalPolynomial.from_string("(5 - 6*x)/(1 + x)")
    b = RationalPolynomial.from_string("(4 - 3*x^2)/(1 - x)")
    c = RationalPolynomial.from_string("(1 - 15*x + 9*x^2 + 3*x^3)/(1 - x^2)")
    assert a - b == c
	
def test_rat_polynomial_subtraction2():
    a = RationalPolynomial.from_string("(-x^3 + 8*x + 3)/(3 - x)")
    b = RationalPolynomial.from_string("(6*x^3 + 25*x^2 + 2*x + 9)/(7*x - 1)")
    c = RationalPolynomial.from_string("(x^3 - 5*x^2 + 2*x - 10)/(7*x - 1)")
    assert a - b == c
	
def test_rat_polynomial_subtraction3():
    a = RationalPolynomial.from_string("(-3 - x^2 + 2*x^3)/(19*x^5)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(-19*x^5)")
    c = RationalPolynomial.from_string("(-2 + 6*x^2 + 2*x^3)/(19*x^5)")
    assert a - b == c
	
def test_rat_polynomial_subtraction4():
    a = RationalPolynomial.from_string("(0+0*x^3)/(-3 -x -7*x^2)")
    b = RationalPolynomial.from_string("(1 + 7*x^2)/(9 + 3*x + 17*x^2)")
    c = RationalPolynomial.from_string("(1 + 7*x^2)/(-9 - 3*x - 17*x^2)")
    assert a - b == c
	
def test_rat_polynomial_multiplication():
    a = RationalPolynomial.from_string("(4)/(5)")
    b = RationalPolynomial.from_string("(2 - x + 3*x^2)/(1 + 3*x)")
    c = RationalPolynomial.from_string("(8 - 4*x + 12*x^2)/(5 + 15*x)")
    assert a * b == c
	
def test_rat_polynomial_multiplication2():
    a = RationalPolynomial.from_string("(5 + 15*x)/(8 - 4*x + 12*x^2)")
    b = RationalPolynomial.from_string("(2 - x + 3*x^2)/(1 + 3*x)")
    c = RationalPolynomial.from_string("(5)/(4)")
    assert a * b == c
	
def test_rat_polynomial_multiplication3():
    a = RationalPolynomial.from_string("(3 - 2*x^2 + x^3)/(3 - 2*x^2 + x^3)")
    b = RationalPolynomial.from_string("(2 - x + 3*x^2)/(19*x^6 - 4)")
    c = RationalPolynomial.from_string("(2 - x + 3*x^2)/(19*x^6 - 4)")
    assert a * b == c	
	
def test_rat_polynomial_multiplication4():
    a = RationalPolynomial.from_string("(0)/(3 - 2*x^2 + x^3)")
    b = RationalPolynomial.from_string("(2 - x)/(3*x^2)")
    c = RationalPolynomial.from_string("(0)/(1)")
    assert a * b == c		
	
def test_rat_polynomial_division():
    a = RationalPolynomial.from_string("(4)/(5)")
    b = RationalPolynomial.from_string("(1 + 3*x)/(2 - x + 3*x^2)")
    c = RationalPolynomial.from_string("(8 - 4*x + 12*x^2)/(5 + 15*x)")
    assert a / b == c
	
def test_rat_polynomial_division2():
    a = RationalPolynomial.from_string("(5 + 15*x)/(8 - 4*x + 12*x^2)")
    b = RationalPolynomial.from_string("(1 + 3*x)/(2 - x + 3*x^2)")
    c = RationalPolynomial.from_string("(5)/(4)")
    assert a / b == c
	
def test_rat_polynomial_division3():
    a = RationalPolynomial.from_string("(3 - 2*x^2 + x^3)/(3 - 2*x^2 + x^3)")
    b = RationalPolynomial.from_string("(19*x^6 - 4)/(2 - x + 3*x^2)")
    c = RationalPolynomial.from_string("(2 - x + 3*x^2)/(19*x^6 - 4)")
    assert a / b == c	
	
def test_rat_polynomial_division4():
    a = RationalPolynomial.from_string("(0)/(3 - 2*x^2 + x^3)")
    b = RationalPolynomial.from_string("(3*x^2)/(2 - x)")
    c = RationalPolynomial.from_string("(0)/(1)")
    assert a / b == c
	
def test_rat_polynomial_division5():
    a = RationalPolynomial.from_string("(3*x^2)/(2 - x)")
    b = RationalPolynomial.from_string("(0)/(3 - 2*x^2 + x^3)")
    c = RationalPolynomial.from_string("(1)/(0)")
    assert a / b != c