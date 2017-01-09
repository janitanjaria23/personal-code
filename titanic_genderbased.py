import csv
import numpy as np

csv_file = csv.reader(open('/var/data/practice_data/titanic_train_data.csv', 'rb'))

header = csv_file.next()
print header
data = []

for row in csv_file:
    data.append(row)

data = np.array(data)

number_passengers = np.size(data[0::, 1].astype(np.float))
number_survived = np.sum(data[0::, 1].astype(np.float))

proportion_survivors = number_survived/number_passengers

count_of_women = data[0::, 4] == 'female'
count_of_men = data[0::, 4] != 'female'

women_on_board = data[count_of_women, 1].astype(np.float)
men_on_board = data[count_of_men, 1].astype(np.float)

proportion_women_survived = np.sum(women_on_board) / np.size(women_on_board)

proportion_men_survived = np.sum(men_on_board) / np.size(men_on_board)

print women_on_board, men_on_board, proportion_women_survived, proportion_men_survived

test_csv_file = csv.reader(open('/var/data/practice_data/titanic_test.csv', 'rb'))

test_csv_header = test_csv_file.next()
print test_csv_header

prediction_file = csv.writer(open('/var/data/practice_data/genderbased_titanic_model.csv', 'wb'))
prediction_file.writerow(['PassengerId', 'Survived'])
for row in test_csv_file:
    if row[3] == 'female':
        prediction_file.writerow([row[0], '1'])
    else:
        prediction_file.writerow([row[0], '0'])
