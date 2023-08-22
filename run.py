import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Try different encodings until you find the correct one
encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'utf-16', 'utf-32']

df = None
for encoding in encodings:
    try:
        df = pd.read_csv('sth.csv', encoding=encoding)
        break
    except (UnicodeDecodeError, pd.errors.ParserError):
        continue

if df is None:
    raise ValueError("Failed to read the CSV file with all attempted encodings.")

# Rest of your code...


df = pd.read_csv('sth.csv')
#print(df)

Y = df['Type']
X = df.drop(['Type','Id'],axis=1)
#print(X)
#print(X.shape)

#print(Y)
#print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#print(X_train.shape)

#====KNN=======
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

knn_cm = confusion_matrix(y_test,knn_pred)
print(knn_cm)

knn_acc = accuracy_score(y_test,knn_pred)
print('Accuracy of our model is equal'+ str (round(knn_acc,2)) + '%')

pickle.dump(knn, open('knn_classifier.pkl','wb'))

knn_model = pickle.load(open('knn_classifier.pkl', 'rb'))
pred = knn_model.predict([[2600,0.0004,0.096,17.4]])
print(pred[0])



