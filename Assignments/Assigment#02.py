"""                                          Create a class named as DataBaseManagment contains:
1. Add a method to add user data in DataBase.txt
2. Add a method   to delete user data by email from DataBase.txt
3. Add __repr__ Magic method to get email as a parameter and return the detail of that user
4. Add a method to get all users in a list from DataBase.txt file (hint: returns List of user details) """

class DatabaseManagement:
    def __init__(self):
        self.file_name="Database.txt"

    # method to add user data to the database
    def add_user(self,name,email,phone):
        user_data=f"{name},{email},{phone}\n"
        with open(self.file_name,"a") as file:
            file.write(user_data)
        
    # method to delete the user data
    def delete_user(self,email):
        with open(self.file_name,"r") as file:
            lines=file.readlines()
        
        new_lines=[]
        found=False
        for line in lines:
            parts=line.strip().split(",")
            if parts[1]!=email:
                new_lines.append(line)
            else:
                found=True
            
        if found:
            with open(self.file_name,"w") as file:
                file.writelines(new_lines)
            print(f"User with email: {email} deleted.")
        else:
            print("User not found!")

    # magic method to get user details by email
    def __repr__(self,email=None):
        if email:
            with open(self.file_name,"r") as file:
                for line in file:
                    name, user_email,phone=line.strip().split(",")
                    if user_email==email:
                        return f"User(Name={name}, Email={user_email}, Phone={phone})"
            return "User not found!"
        return "DatabaseManagement instance"
    
    #method to get al((l users 
    def get_all_users(self):
        with open(self.file_name,"r") as file:
            users=[line.strip() for line in file if line.strip()]
        return users


db=DatabaseManagement()

# Add users
db.add_user("Ahsan Ali","ahsan@gmail.com","111")
db.add_user("Zeeshan","zeeshan@gmail.com","222")

# Get all the users
print(db.get_all_users())

# Delete user
db.delete_user("zeeshan@gmail.com")

print(db.__repr__("ahsan@gmail.com"))