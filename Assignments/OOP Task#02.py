"""                                                                  Task B:
Create a University class that manages a collection of Student objects. Implement methods to add, remove, and display student information."""

class Student:
    def __init__(self,student_id,student_name,department):
        self.student_id=student_id
        self.student_name=student_name
        self.department=department
    
    def __repr__(self):
        return f"Student(ID: {self.student_id}, Name: {self.student_name}, Department: {self.department})"
    
class University:
    def __init__(self,name):
        self.name=name
        self.students=[]
    
    def add_student(self,student):
        if isinstance(student,Student):
            self.students.append(student)
        else:
            raise TypeError("Sorry, This is not student object")
    
    def remove_student(self,student_id):
        for i, student in enumerate(self.students):
            if student.student_id==student_id:
                del self.students[i]
                return True
        return False
        
    def display_student(self):
        print(f"Students at {self.name}: ")
        for student in self.students:
            print(f"- {student}")

university=University("Mehran University")
university.add_student(Student(101,"Ahsan Ali","Software Engineering"))
university.add_student(Student(102,"Nouman","Electrical Engineering"))
university.display_student()
university.remove_student(102)
university.display_student()