import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV 
from ydata_profiling import ProfileReport
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
# prepare data
df = pd.read_csv('kc_house_data.csv')

# profile = ProfileReport(df, title='report of house price', explorative=True)
# profile.to_file('Visualize_original_data.html')

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month.astype('category')
df['year'] = df['date'].dt.year.astype('category')
df = df.drop('date',axis=1)
df = df.drop('id', axis=1)
df = df.drop('zipcode',axis=1)

#split data
target = 'price'
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# preprocessing data
num_col = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot'
           ,'floors','grade','sqft_above','sqft_basement']
category_columns = ['month', 'year','waterfront','condition','view']

num_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', missing_values=np.nan)),
    ('scaler', MinMaxScaler()),
])


category_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transform, num_col),
        ('cat', category_transform, category_columns),
    ]
)


X_train = preprocessor.fit_transform(X_train,y_train)
X_test = preprocessor.transform(X_test)

# check the weights
checkpoint_path = "model_weight/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model
def create_model():
    model = Sequential()
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mse', metrics=[MeanSquaredError()])
    return model

model = create_model()

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test.values),
                    epochs=400, batch_size=130,
                    callbacks=[cp_callback]
                    )
model.save('result/model.h5')


import matplotlib.pyplot as plt
import seaborn as sns
losses = pd.DataFrame(model.history.history)

plt.figure(figsize=(15,5))
sns.lineplot(data=losses,lw=3)
plt.xlabel('Epochs')
plt.ylabel('')
plt.title('Training Loss per Epoch')
sns.despine()
plt.savefig('Visualize.png')


# predictions on the test set
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)

with open('result/Loss.txt', 'w') as f:
    f.write('MAE:  {} \n'.format(mean_absolute_error(y_test,predictions)))
    f.write('MSE:  {} \n'.format(mean_squared_error(y_test,predictions)))
    f.write('RMSE: {} \n' .format(np.sqrt(mean_squared_error(y_test,predictions))))
    f.write('Variance Regression Score: {}'.format(explained_variance_score(y_test,predictions)))


with open('result/prediction.txt', 'w') as f:
    for i,j in zip(predictions, y_test):
        f.write('predicted price: {} , true price: {} \n'.format(i,j))
