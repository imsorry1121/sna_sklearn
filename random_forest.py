from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from numpy import genfromtxt

# forest = RandomForestClassfier
# delete title
train_data = genfromtxt('train_feature.csv',delimiter=',',skip_header=1)
print(train_data)
test_data = genfromtxt('test_feature.csv',delimiter=',',skip_header=1)
x = train_data[:,3:]
y = train_data[:,0:1]
rf = RandomForestClassifier(n_estimators = 10)
rf_model = rf.fit(x,y)
output = rf_model.predict(test_data)
print(output)

