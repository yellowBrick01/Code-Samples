user_numbers = [] # To create  list to hold the user's numbers
i = 1 # This is used to create an infinte loop
user_message = "" #To check if the string for "complete" and to record the value
total = 0 # To get the sum of user values
average = 0 # To record the average of the user values

while i == 1: # This will prompt the user until they enter "complete"
    user_message = input("Enter a number or complete to end this program: ")
    if user_message == "complete": # The if statemnt will break the loop if "complete" is entered
        break
    else:
        try:
          user_numbers.append(int(user_message))
        except ValueError:
            print ("Not a valid number. Please enter something like: 1, 2, or 3")
            continue 
    
print("You have entered",len(user_numbers),"values.")

print("Those values were:")
for x in range(len(user_numbers)):
    print(user_numbers[x])
    
for x in range(len(user_numbers)):
    total += user_numbers[x]

average = total / len(user_numbers)
print("Here is the total of the values entered: ", total)
print("Here is the average of the values entered: ", average)