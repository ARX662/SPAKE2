The process to run the program:
The program is built with JAVA corretto-17 jdk and jpair.jar as dependencies.
First Add Jpair.jar as a dependency for the project if not done so already.
Then run the Run All class which then runs the program, and the user will be prompted a password to enter, Enter the password and then the program will compute and say if the session is secure or not.
Tests are a series of print statements, they can be uncommented and checked if the system sends and receives the same information  for the admin and the client.
After the program is executed run DeleteFiles class before re-running the program again as initially the program checks if the password file and others are created or not.
If the delete files do not dynamically take the current directory please add the absolute path for the  project folder.
Uncomment the print statements at the end of the code to check for the correctness of the output in the client and admin class and uncomment System.nanoTime() in the run-all class to find execution time.\\
The inner SPAKE2 folder has all the runnable code.
Click on the inner SPAKE2 folder and  the bin has jpair.jar.
the src folder has the runnable code.
files created are saved in the outer SPAKE2 folder.
if any exception is thrown run delete files and re-run the program. caused due to an error in the packages and library used.