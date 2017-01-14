import csv
import numpy as np
import seaborn as sns

csv_file = csv.reader(open('/var/data/practice_data/titanic_train_data.csv', 'rb'))

header = csv_file.next()
print header
data = []

for row in csv_file:
    data.append(row)

data = np.array(data)

fare_ceil = 40
data[data[0::, 9].astype(np.float) >= fare_ceil, 9] = fare_ceil - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceil / fare_bracket_size

number_of_classes = 3

number_of_classes = len(np.unique(data[0::, 2]))

survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(0, number_of_classes):
    for j in range(0, number_of_price_brackets):
        count_of_women_only = data[
            (data[0::, 4] == 'female') & (data[0::, 2].astype(np.float) == i + 1) & (data[0::, 9].astype(np.float) >= j * fare_bracket_size) & (
            data[0::, 9].astype(np.float) <= (j + 1) * fare_bracket_size), 1]

        count_of_men_only = data[
            (data[0::, 4] != 'female') & (data[0::, 2].astype(np.float) == i + 1) & (data[0::, 9].astype(np.float) >= j * fare_bracket_size) & (
                data[0::, 9].astype(np.float) <= (j + 1) * fare_bracket_size), 1]
        # print count_of_men_only, count_of_women_only
        print np.sum(count_of_women_only.astype(np.float))
        survival_table[0, i, j] = np.mean(count_of_women_only.astype(np.float))
        survival_table[1, i, j] = np.mean(count_of_men_only.astype(np.float))
        survival_table[survival_table != survival_table] = 0.

print survival_table[0, :, :]

survival_table[survival_table >= 0.5] = 1
survival_table[survival_table < 0.5] = 0

print survival_table

test_csv_file = csv.reader(open('/var/data/practice_data/titanic_test.csv', 'rb'))

test_csv_header = test_csv_file.next()
print test_csv_header

prediction_file = csv.writer(open('/var/data/practice_data/genderbasedcomposite_titanic_model.csv', 'wb'))
prediction_file.writerow(['PassengerId', 'Survived'])


for row in test_csv_file:
    for j in xrange(number_of_price_brackets):
        # If there is no fare then place the price of the ticket according to class
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceil:
            bin_fare = number_of_price_brackets - 1
            break

        if j*fare_bracket_size <= row[8] < (j+1)*fare_bracket_size:
            bin_fare = j
            break
    if row[3] == 'female':
        prediction_file.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, bin_fare ])])
    else:
        prediction_file.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, bin_fare])])




