user_details=[
    ["ahsan","ahsanali@gmail.com","abc123"],
    ["zeeshan","zeeshanali@gmail.com","def321"],
    ["ameet","ameet@gmail.com","xyz123"]
]

while True:
    status=False
    user_email=input("Enter email address: ")
    for user in user_details:
        if user_email==user[1]:
            status=True
            print("User name is: ",user[0])
            break
    else:
        print("User not found!")
    if status:
        break




