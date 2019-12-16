# Change 'newton.csv' in with appropriate csv you want to fill empty rows with
# Change 'filled_list.csv' to any name you like, this is the name of the ouput csv you want


import csv
from datetime import datetime
from datetime import timedelta

input_csv = 'newton.csv'
output_csv_name = 'filled_rows.csv'


my_list_changi = []
my_list_other = []

my_list_filled = []

my_dict = {}

with open('changi.csv', newline='') as csvfile:
	tempreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in tempreader:
		row[3] = None
		my_list_changi.append(row)

print(my_list_changi)

with open(input_csv, newline='') as csvfile:
	tempreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in tempreader:
		for data in my_list_changi:
			if ((data[0] == row[0]) and (data[1] == row[1]) and (data[2] == row[2])):
				data[3] = row[3]
 

# print(my_list_changi)

with open(output_csv_name, 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	for row in my_list_changi:
		wr.writerow(row)

""""
date_str = '1998-04-01 01:00:00'
datetime_object = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
# print(datetime_object + timedelta(hours=1))
for i in range(32904):
	my_list_filled.append([str(datetime_object + timedelta(hours=i))])




print(my_list_filled[0])
print(my_list_filled[-1])

avg = 29766.4274075875 #Using excel
stdev = 5849.76995358656 #Using excel

normalization_mean = 30000

for index in range(1, len(my_list)):
	my_dict[my_list[index][0]] = my_list[index][1]

for row in my_list_filled:
	if row[0] in my_dict:
		row.append(my_dict[row[0]])
	else:
		row.append(avg)
print(my_list_filled)

with open('filled_list.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	for row in my_list_filled:
		wr.writerow(row)
# print(datetime_object + timedelta(hours = 32897)) 
"""

