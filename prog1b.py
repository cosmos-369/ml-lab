import csv  # Import the CSV module to handle reading from CSV files

num_attributes = 6  # Number of attributes in each training instance (excluding the target/output)

a = []  # Initialize an empty list to hold the dataset

print("\n The given training data set \n")

csvfile = open('./data/finds.csv', 'r')  # Open the 'tennis.csv' file in read mode

reader = csv.reader(csvfile)  # Create a CSV reader object to parse the file

# Read each row from the CSV file
for row in reader:
    a.append(row)  # Add the row (training instance) to the dataset list
    print(row)  # Print the row to show the input training data

print("The initial values of hypothesis ")

hypothesis = ['0'] * num_attributes  # Initialize the hypothesis with the most specific values (all '0')
print(hypothesis)  # Print the initial hypothesis

# Set the initial hypothesis to the first training instance
for j in range(0, num_attributes):
    hypothesis[j] = a[0][j]  # Copy each attribute value from the first training instance

# Iterate through all training instances in the dataset
for i in range(0, len(a)):
    # Consider only positive instances (target label is 'yes')
    if a[i][num_attributes] == 'yes':
        # Compare the instance with the current hypothesis
        for j in range(0, num_attributes):
            # If attribute value differs from the hypothesis, generalize that attribute to '?'
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                # If the value matches, retain it
                hypothesis[j] = a[i][j]
    # Print hypothesis after processing each instance
    print("For training instance no:", i, "the hypothesis is", hypothesis)

# Print the final hypothesis after processing all training instances
print("The maximally specific hypothesis is ", hypothesis)
