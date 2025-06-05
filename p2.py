import csv 

file = "./data/PlayTennis.csv" 
a = [] 
with open(file) as csvfile:
    for row in csv.reader(csvfile): 
        a.append(row) 
        print(row) 

print("\n--- Running Candidate Elimination ---\n")

# initialize s and g 
num_attr = len(a[0]) - 1
s = ['0'] * num_attr 
g = ['?'] * num_attr 

# initialize s with first positive instance
for i, attr in enumerate(a[1][1:]): 
    s[i] = attr

temp = []  # general hypotheses (G)

for idx, row in enumerate(a[1:], 1): 
    decision = row[0]
    row = row[1:] 
    print(f"Instance {idx}: {row}, Label: {decision}")

    if decision == "Yes": 
        for i in range(num_attr): 
            if row[i] != s[i]: 
                s[i] = '?' 

        temp = [h for h in temp if all(h[j] == '?' or h[j] == s[j] for j in range(num_attr))]
    else: 
        new_g_hypo = []
        for i in range(num_attr): 
            if row[i] != s[i] and s[i] != '?': 
                new_g = g[:] 
                new_g[i] = s[i] 
                new_g_hypo.append(new_g) 
        temp.extend(new_g_hypo)

    print("S:", s)
    print("G:", temp if temp else [g])
    print()
