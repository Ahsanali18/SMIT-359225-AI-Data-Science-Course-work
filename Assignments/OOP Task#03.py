"""                                                                 Task C:
Create a Person class with attributes name and age. Create a Student class that  inherits from Person and adds attributes 
roll_number and grades."""

class Person:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    
    def display_information(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")

class Student(Person):
    def __init__(self,name,age,roll_number):
        super().__init__(name,age)
        self.roll_number=roll_number
        self.grades=[]

    def add_grade(self,grade):
        self.grades.append(grade)

    def display_information(self):
        super().display_information()
        print(f"Roll Number: {self.roll_number}")
        print(f"Grades: {','.join(map(str,self.grades))}")
    
student=Student("Ahsan Ali",19,"Sw00045")
student.add_grade(90)
student.add_grade(85)
student.display_information()