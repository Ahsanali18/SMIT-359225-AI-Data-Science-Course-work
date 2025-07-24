#input() function always takes --> string as an input(you can typecast in your desired datatype)

#Task: Write a program to take an integer as an input from the user and print if the integer is even or odd

a = int(input("Enter an integer "))
if a%2 == 0:
    print("It's even")
else: 
    print("It's odd")
