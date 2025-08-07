"""                                         Weekly Task
Store data of atleast 10 users (In dictionary) Data=contains all the values related to user info- also password 
Email=find email through Loop
if email found ask for password
if pass=correct then print user details 
Make function for searching process
Make use of exception handling too"""

# Solution

# Storing user details into list of dictionaries
user_details=[
    {"name":"ahsan","email":"ahsan@gmail.com","age":19,"password":"ahsan"},
    {"name":"sachin","email":"sachin@gmail.com","age":19,"password":"sachin"},
    {"name":"ameet","email":"ameet@gmail.com","age":20,"password":"ameet"},
    {"name":"basit","email":"basit@gmail.com","age":20,"password":"basit"},
    {"name":"dhani bux","email":"bux@gmail.com","age":22,"password":"dhanibux"},
    {"name":"husnaak","email":"husnaak@gmail.com","age":21,"password":"husnaak"},
    {"name":"majid","email":"majid@gmail.com","age":19,"password":"majid"},
    {"name":"raza","email":"raza@gmail.com","age":20,"password":"raza"},
    {"name":"jibran","email":"jibran@gmail.com","age":20,"password":"jibran"},
    {"name":"soorat","email":"soorat@gmail.com","age":20,"password":"soorat"},
]

# Search for user using email address
def search_user(email_address):
        for user in user_details:
            if user["email"]==email_address:
                password_input=input("Enter password: ")
                if password_input==user["password"]:
                    print("User found(",end=" ")
                    print("Name: ",user["name"],",",end=" ")
                    print("Email: ",user["email"],",",end=" ")
                    print("Age: ",user["age"],", ",end=" ")
                    return
                else:
                    print("Incorrect password")
                    return
        print("Email not found!")

# Main program
try:
    email_input = input("Enter email: ")
    search_user(email_input)
except Exception as e:
    print("Error:", e)
