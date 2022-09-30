import math
import numpy as np

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
        string = string.replace(")","")     #Remove parenthesis
        string = string.replace("(","")
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
            if np.array_equal(self.coeffs,np.array([0],dtype=int)):
                string="0"
            elif self.coeffs[i]!=0 and string=="":
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
        if np.array_equal(coeffs,np.zeros(len(coeffs), dtype=int)):
            self.coeffs = np.array([0], dtype=int)
        elif coeffs[-1]==0:
            i=len(coeffs)-1
            flag=0
            while flag==0:
                if coeffs[i]!=0:
                    flag=1
                else:
                    i-=1
            self.coeffs=coeffs[:i+1]
        self.order = len(self.coeffs)-1
            
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
    
    def __truediv__(self, other):
        return RationalPolynomial(self,other)
    
    def __eq__(self, other):
        return np.array_equal(self.coeffs,other.coeffs)
    
class RationalPolynomial:
    
    def __init__(self, Numerator, Denominator):
        self.Numerator = Numerator
        self.Denominator = Denominator
        self._reduce()
    
    @staticmethod
    def from_string(string):
        Numerator   = Polynomial.from_string(string.split("/")[0])
        Denominator = Polynomial.from_string(string.split("/")[1])
        return RationalPolynomial(Numerator, Denominator)
    
    def __repr__(self):
        string = "("+str(self.Numerator) + ")/(" + str(self.Denominator) + ")"
        return string
    
    def __add__(self, other):
        Numerator = ( (self.Numerator * other.Denominator)
                    + (other.Numerator * self.Denominator))
        Denominator = self.Denominator * other.Denominator
        return RationalPolynomial(Numerator, Denominator)
    
    def _reduce(self):
        #Setup relavent variables
        Numerator = self.Numerator
        Denominator = self.Denominator
        Ncoeffs = Numerator.coeffs
        Dcoeffs = Denominator.coeffs
        NLead = Ncoeffs[len(Ncoeffs)-1]
        DLead = Dcoeffs[len(Dcoeffs)-1]
        #print("Numerator: ", Numerator)
        #print("Denominator: ", Denominator)
        #print("Ncoeffs: ", Ncoeffs)
        #print("Dcoeffs: ", Dcoeffs)
        #print("NLead: ", NLead)
        #print("DLead: ", DLead)
        
        #In the case of equal numerator and denominator
        if np.array_equal(Ncoeffs,Dcoeffs):
            self.Numerator = Polynomial(np.array([1], dtype=int))
            self.Denominator = Polynomial(np.array([1], dtype=int))
        
        #In the case of zero numerator
        elif np.array_equal(Ncoeffs,np.array([0], dtype=int)):
            self.Denominator = Polynomial(np.array([1], dtype=int))
        
        #Cancel common terms if they exist
        else:
            NumRoots = np.sort_complex(np.roots(np.flip(Ncoeffs)))
            DenRoots = np.sort_complex(np.roots(np.flip(Dcoeffs)))
            CommonRoots = np.zeros(max(len(NumRoots),len(DenRoots)), dtype=complex)
                    
            #print("initial NumRoots: ", NumRoots)
            #print("initial DenRoots: ", DenRoots)
            #print("initial Common Roots: ", CommonRoots)
            
            i=0
            k=0
            NumAccounted=np.zeros(len(NumRoots))
            DenAccounted=np.zeros(len(DenRoots))
            while i<len(NumRoots):
                j=0
                while j<len(DenRoots):
                    if NumAccounted[i]==0 and DenAccounted[j]==[0]:
                        if np.isclose(NumRoots[i],DenRoots[j], 1e-5):
                            #print(k)
                            CommonRoots[k]=(NumRoots[i]+DenRoots[j])/2
                            NumAccounted[i]=k+1
                            DenAccounted[j]=k+1
                            #print("NumAccounted: ",NumAccounted)
                            #print("DenAccounted: ",DenAccounted)
                            k+=1
                            break
                    j+=1
                i+=1
            CommonRoots = CommonRoots[:k]
            #CommonRoots = np.intersect1d(NumRoots,DenRoots)          #Already sorted
            
            while not np.array_equal(CommonRoots,np.array([])):
                for k in CommonRoots:
                    N_rootLoc = np.where(np.isclose(NumRoots,k,1e-5))[0]
                    D_rootLoc = np.where(np.isclose(DenRoots,k,1e-5))[0]
                    
                    NumRoots = np.delete(NumRoots,N_rootLoc[0])
                    DenRoots = np.delete(DenRoots,D_rootLoc[0])
                    CommonRoots = np.intersect1d(NumRoots,DenRoots)
            #print("Final NumRoots: ", NumRoots, type(NumRoots))
            #print("Final DenRoots: ", DenRoots, type(DenRoots))

            if np.array_equal(NumRoots,np.array([])):
                Ncoeffs = NLead*np.array([1], dtype=int)
            else:
                Ncoeffs = np.rint(float(NLead)*np.flip(np.poly(NumRoots))).astype(int)
            if np.array_equal(DenRoots,np.array([])):
                Dcoeffs = DLead*np.array([1], dtype=int)
            else:
                Dcoeffs = np.rint(float(DLead)*np.flip(np.poly(DenRoots))).astype(int)
            
            #print("semiFinal Ncoeffs: ", Ncoeffs, type(NumRoots))
            #print("semiFinal Dcoeffs: ", Dcoeffs, type(DenRoots))
            
            #Fix sign of leading coefficients and reduce coeffs
            gcd = math.gcd(*Ncoeffs, *Dcoeffs)
            #print("gcd = ", gcd)
            Ncoeffs = Ncoeffs // gcd
            Dcoeffs = Dcoeffs // gcd
            NLead = Ncoeffs[len(Ncoeffs)-1]
            DLead = Dcoeffs[len(Dcoeffs)-1]
            if DLead < 0:
                Ncoeffs *= -1
                Dcoeffs *= -1

            #print("Final Ncoeffs: ", Ncoeffs)
            #print("Final Dcoeffs: ", Dcoeffs)
            
            self.Numerator = Polynomial(Ncoeffs)
            self.Denominator = Polynomial(Dcoeffs)
            
    def __neg__(self):
        Numerator = Polynomial.from_string("-1")*self.Numerator
        return RationalPolynomial(Numerator, self.Denominator)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        return RationalPolynomial(self.Numerator*other.Numerator,
                        self.Denominator*other.Denominator)
    
    def __truediv__(self, other):
        return RationalPolynomial(self.Numerator*other.Denominator,
                        self.Denominator*other.Numerator)
    
    def __eq__(self, other):
        if self.Numerator == other.Numerator:
            if self.Denominator == other.Denominator:
                return True
        return False