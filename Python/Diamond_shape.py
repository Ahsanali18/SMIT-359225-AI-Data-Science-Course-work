# Task: Make the Diamond like shape
"""
                                     *
                                    * *
                                   * * *
                                  * * * *
                                 * * * * *
                                 * * * * *
                                  * * * *
                                   * * *
                                    * *
                                     *
"""
for row in range(0,5):
    for space in range(6,row+1,-1):
        print(" ",end="")
    for star in range(0,row+1):
        print("* ",end="")
    print()


for row in range(5,-1,-1):
    for star in range(5,row,-1):
        print(" ",end="")
    for space in range(0,row+1):
        print("* ",end="")
    print()