"""                                                         Task A:
Create a class named BankAccount add some properties and magic methods like __add__  by which if i need to add some amount to the 
instance of class BankAccount the amount should be incremented in self.balance property."""

class BankAccount:
    def __init__(self, account_holder, initial_balance):
        self.account_holder=account_holder
        self.balance=initial_balance

    def __add__(self,amount):
        if isinstance(amount,(int,float)):
            self.balance+=amount
            return self
        else:
            raise ValueError("Can only add numeric values to the balance.")
        
    def __repr__(self):
        return f"BankAccount(holder='{self.account_holder}',balance= {self.balance})"
    
    def display_balance(self):
        print(f"Account holder: {self.account_holder}")
        print(f"Current balance: ${self.balance}")


account=BankAccount("Ahsan Ali",5000)
account+1000 #Adding amount 1000
account.display_balance()