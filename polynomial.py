import math
import numpy as np
import re

# Definition of a polynomial is stored in arrays coeffs
# Length of array denotes maximum order, an entry denotes the coefficient, the index is the order of the term.
# -> entries may be positive or negative
# Coefficients tested will only be integers. No irrational coeffiecients.
class Polynomial:
    
    def __init__(self, coeffs):
        self.order = len(coeffs)-1
        self.coeffs=coeffs
        self._reduce()
    
    @staticmethod
    def from_string(string):

        #Clean string
        string = string.replace(" ","")     #Remove spaces
        i=0
        while i<len(string)-1:
            if string[i:i+2]=="+-":         #replaces all instances of "+-" and "-+" with "-"
                string=string[:i]+"-"+string[i+2:]
                i=0
            elif string[i:i+2]=="-+":
                string=string[:i]+"-"+string[i+2:]
                i=0
            else:
                i+=1    
        i=0
        while i<len(string)-1:
            if string[i:i+2]=="--":         #remove repeated subtraction
                string=string[:i]+"+"+string[i+2:]
                i=0
            else:
                i+=1
        string = string.replace("-","+-")   #place addition before each subtraction
        i=0
        while i<len(string)-1:
            if string[i:i+2]=="++":         #remove repeated addition
                string=string[:i]+"+"+string[i+2:]
                i=0
            else:
                i+=1
        if string[0]=="+":                  #remove addition symbol if it is the first character
            string=string[1:]
        if string[-1]=="+":                 #remove addition symbol if it is the first character
            string=string[:-1]
        string = string.replace("+0+","+")  #remove addition of 0
        if string[:2]=="0+":
            string=string[2:]
        if string[-2:]=="+0":
            string=string[:-2]
        string = string.replace("*x^",",")  #add comma to seperate coeffs and powers "1,2+3,4 etc"
        string = string.replace("*x",",1")  #for the case where no explicit exponent
        string = string.replace("x^","1,")  #for the case of no coefficient
        string = string.replace("x","1,1")  #for the case where no exponent and no coefficient
        terms = string.split("+")           #string array of terms
        
        #Deal with order 0 terms
        i=0
        while i<len(terms):
            if "," in terms[i]:
                i+=1
            else:
                terms[i]=terms[i]+",0"
                i+=1
        
        #Define arrays for splitting and organizing terms
        Nterms = len(terms)
        powers = np.zeros(Nterms, dtype=int)#unordered powers
        coughs = np.zeros(Nterms, dtype=int)#unordered coefficients
        
        i=0
        while i<Nterms:
            if terms[i][-1]==",":           #for the case the term was a constant
                terms[i]=terms[i]+"0"
            coughs[i] = terms[i].split(",")[0]
            powers[i] = terms[i].split(",")[1] 
            i+=1
        #Reorganize data into the final polynomial datatype
        order = max(powers)
        coeffs = np.zeros(order+1, dtype=int)
        i=0
        while i<len(powers):
            coeffs[powers[i]] += coughs[i]
            i+=1
        return Polynomial(coeffs)
    
    def __repr__(self):
        i=0
        string=""
        while i<self.order+1:
            if self.coeffs[i]!=0 and string=="":
                if i==0:
                    string = str(self.coeffs[0])
                elif i==1:
                    string = str(self.coeffs[1])+"x"
                else:
                    string = str(self.coeffs[i])+"x^"+str(i)
            elif self.coeffs[i]!=0:
                if i==1:
                    string = string+"+"+str(self.coeffs[1])+"x"
                else:
                    string = string+"+"+str(self.coeffs[i])+"x^"+str(i)
            i+=1
        string = string.replace("+-"," - ")
        string = string.replace("+"," + ")
        return string
    
    def __add__(self, other):
        Cself=self.coeffs
        Cother=other.coeffs
        maxOrder = max(len(Cself),len(Cother))
        if len(Cother)!=maxOrder:
            Cother=np.concatenate((Cother,np.zeros(maxOrder-len(Cother),dtype=int)), axis=None)
        elif len(self.coeffs)!=maxOrder:
            Cself=np.concatenate((Cself,np.zeros(maxOrder-len(Cself),dtype=int)), axis=None)

        coeffs=np.add(Cself,Cother)
        return Polynomial(coeffs)
    
    def _reduce(self):
        coeffs=self.coeffs
        if coeffs[-1]==0:
            i=len(coeffs)-1
            flag=0
            while flag==0:
                if coeffs[i]!=0:
                    flag=1
                else:
                    i-=1
            self.coeffs=coeffs[:i+1]
            
    def __mul__(self, other):
        return Polynomial(np.flip(np.polymul(np.flip(self.coeffs),np.flip(other.coeffs))))
    
    def __sub__(self, other):
        Cself=self.coeffs
        Cother=other.coeffs
        maxOrder = max(len(Cself),len(Cother))
        if len(Cother)!=maxOrder:
            Cother=np.concatenate((Cother,np.zeros(maxOrder-len(Cother),dtype=int)), axis=None)
        elif len(self.coeffs)!=maxOrder:
            Cself=np.concatenate((Cself,np.zeros(maxOrder-len(Cself),dtype=int)), axis=None)

        coeffs=np.subtract(Cself,Cother)
        return Polynomial(coeffs)
    
    def __eq__(self, other):
        return np.array_equal(self.coeffs,other.coeffs)
    
    #def __truediv__(self, other):
    #    return Fraction(self.numerator*other.denominator,
    #                    self.denominator*other.numerator)