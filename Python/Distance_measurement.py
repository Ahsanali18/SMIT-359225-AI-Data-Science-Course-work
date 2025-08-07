import math

# d= √(x2-x1)² + (y2-y1)²

def find_distance(p1,p2):
    dx=p2[0]-p1[0]
    dy=p2[1]-p1[1]

    return math.sqrt(dx**2+dy**2)

x1=float(input("Enter x1: "))
y1=float(input("Enter y1: "))
x2=float(input("Enter x2: "))
y2=float(input("Enter y2: "))

p1=(x1,y1)
p2=(x2,y2)

# Method-1: By using math.dist method
print("Distance: ",math.dist(p1,p2))

# Method-2: Using our own Method
print("Distance (custom): ",find_distance(p1,p2))

    


