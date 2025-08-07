# Create a function to find the factorial of number using recursion
"""def factorial(number):
    if number==0 or number==1:
        return 1
    else:
        return number*factorial(number-1)

number=int(input("Enter number: "))
print("Factorial of",number,"is:",factorial(number))"""


# Create a function to find the factorial of number using loop
def factorial(number):
    fact=1
    for i in range(number,0,-1):
        fact=fact*i
    return fact

number=int(input("Enter number: "))
print("Factorial of",number,"is:",factorial(number))



