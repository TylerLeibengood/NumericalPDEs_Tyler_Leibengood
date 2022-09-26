
import numpy as np
import numbers

class C:
    
    def __init__(self, data):
        self.data = np.array(data)
    
    def __repr__(self):
        return "%s + %s i" %(self.data[0], self.data[1])
    
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    
    def __add__(self, other):
        return C( self.data + other.data )
    
    def __neg__(self):
        return C( -self.data )
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance(other, C):
            return C( (self.data[0]*other.data[0] - self.data[1]*other.data[1],
                       self.data[0]*other.data[1] + self.data[1]*other.data[0]) )
        elif isinstance(other, numbers.Number):
            return C( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return C( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")    

