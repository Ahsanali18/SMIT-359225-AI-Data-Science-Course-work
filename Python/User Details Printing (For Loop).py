user_details=[
    ["ahsan","ahsanali@gmail.com","abc123"],
    ["zeeshan","zeeshanali@gmail.com","def321"],
    ["ameet","ameet@gmail.com","xyz123"]
]

# for user in user_details:
#     print(user)

user_name=input("Enter user name to display: ")

for user in user_details:
    if(user_name==user[0]):
        print(user_name)
else:
    print("User not found!")