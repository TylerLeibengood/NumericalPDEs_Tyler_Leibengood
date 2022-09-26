import math

class Fraction:
    
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()
    
    @staticmethod
    def from_string(string):
        numerator   = int(string.split("/")[0])
        denominator = int(string.split("/")[1])
        return Fraction(numerator, denominator)
    
    def __repr__(self):
        string = str(self.numerator) + " / " + str(self.denominator)
        return string
    
    def __add__(self, other):
        numerator = ( self.numerator * other.denominator
                   + other.numerator *  self.denominator)
        denominator = self.denominator * other.denominator
        return Fraction(numerator, denominator)
    
    def _reduce(self):
        gcd = math.gcd(self.numerator, self.denominator)
        self.numerator   = self.numerator   // gcd
        self.denominator = self.denominator // gcd
        if self.denominator < 0:
            self.numerator *= -1
            self.denominator *= -1
            
    def __neg__(self):
        numerator = -1*self.numerator
        return Fraction(numerator, self.denominator)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        return Fraction(self.numerator*other.numerator,
                        self.denominator*other.denominator)
    
    def __truediv__(self, other):
        return Fraction(self.numerator*other.denominator,
                        self.denominator*other.numerator)
    
    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False
