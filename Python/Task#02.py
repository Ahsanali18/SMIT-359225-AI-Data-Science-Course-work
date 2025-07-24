"""
Task: Write a program to store user information in variables including password. 
Take email and password as an input from the user if email and password is matched than print complete details in the given format 
else print password not matched"""


original_password = "abc123"
orignal_email = "ahsan@gmail.com"
name = "Ahsan Ali"
phone= "03XXXXXXXXX"
age = 20
address = "Qasimabad"

email = input("Enter email address: ")
password = input("Enter password: ")

if password == original_password and email == orignal_email:
    print("Passowrd & Email matched!😊, Here are your details: ")
    print("Name: ",name)
    print("Phone: ",phone)
    print("Age: ",age)
    print("Email: ",email)
    print("Address: ",address)
else :
    print('Invalid email or password')    
