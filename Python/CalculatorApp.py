
def addition(operand_1,operand_2):
    return operand_1+operand_2

def subtraction(operand_1,operand_2):
    return operand_1-operand_2

def multiplication(operand_1,operand_2):
    return operand_1*operand_2

def division(operand_1,operand_2):
        if operand_2==0:
            print("Divison by 0 is not possible.")
        return operand_1/operand_2
    

print("Calculator App")
print("1. Add(➕)")
print("2. Subtract")
print("3. Multiplication ✖")
print("4. Division➗")

choice=int(input("Enter your choice: "))

if choice==1:
    # operand_1=int(input("Enter value-1: "))
    # operand_2=int(input("Enter value-2: "))
    # print("Additoin is: ",addition(operand_1,operand_2))
    print("Result is: ",addition(int(input("Enter value-1: ")),int(input("Enter value-2: "))))

elif choice==2:
    # operand_1=int(input("Enter value-1: "))
    # operand_2=int(input("Enter value-2: "))
    # print("Subtraction is: ",subtraction(operand_1,operand_2))
    print("Result is: ",subtraction(int(input("Enter value-1: ")),int(input("Enter value-2: "))))
    
elif choice==3:
    # operand_1=int(input("Enter value-1: "))
    # operand_2=int(input("Enter value-2: "))
    # print("Multiplication is: ",multiplication(operand_1,operand_2))
    print("Result is: ",multiplication(int(input("Enter value-1: ")),int(input("Enter value-2: "))))

elif choice==4:
    # operand_1=int(input("Enter value-1: "))
    # operand_2=int(input("Enter value-2: "))
    # print("Division is: ",division(operand_1,operand_2))
    print("Result is: ",division(int(input("Enter value-1: ")),int(input("Enter value-2: "))))

else:
    print("Invalid choice!")