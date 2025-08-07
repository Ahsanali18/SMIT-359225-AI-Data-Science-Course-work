def area(length,width):
    return length*width

def volume(length,width,height):
    return length*width*height

print("===== Area and Volume Calculator =====")
print("1. Area")
print("2. Volume")

user_choice=int(input("Please enter your choice: "))

if user_choice==1:
    try:
        length=int(input("Enter length: "))
        width=int(input("Enter width: "))
        print("Area is: ",area(length,width))
    except ValueError as e:
        print("Please enter number only",e)
elif user_choice==2:
    try:
        length=int(input("Enter length: "))
        width=int(input("Enter width: "))
        height=int(input("Enter height: "))
        print("Volume is: ",volume(length,width,height))
    except ValueError as e:
        print("Please enter number only",e)
else:
    print("Invalid input!")


