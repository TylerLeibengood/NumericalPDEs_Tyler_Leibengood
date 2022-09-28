import math

numberString=input("Numbers to add: ")
Numbers=numberString.split(" ")
i=0
while i<len(Numbers):
    Numbers[i] = int(Numbers[i])
    i=i+1
print(sum(Numbers))