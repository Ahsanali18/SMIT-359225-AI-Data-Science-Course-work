"""Functions
      A block of code which is separated from atmosphere just to perform some specific task when it is called.
      it is used to increase reusability and reduce redundancy and repitition.

      syntax:
        def func_name():
          body of
        'def' keyword is used

def my_function():
    while True:
      try: 
        a = int(input("Please enter a number: "))
        if a ==5:
           break
      except ValueError as e:
        print("PLease enter number only",e)

my_function()"""




def add():
    a = 5
    b = 6
    return a+b

a = 10
print(add())       #add will print 11


def add(a=1,b=2):  #default value
    return a+b

print(add(7,8))     #prints sum of requried value --> prints 15
print(add())   #prints sum of default values
