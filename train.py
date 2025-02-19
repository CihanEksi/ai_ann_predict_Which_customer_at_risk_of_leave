import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

## Static variables
file_to_train = "train_excel.csv"
unwanted_columns = ['RowNumber','CustomerId','Surname']
categorical_fields = ['Gender']
ordinal_fields = ['Geography']
target_field = 'Exited'
metrics = ['accuracy']
epochs=100
patience=10
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
label_pkl_file_name = "label_encoder.pkl"
ordinal_pkl_file_name = "ordinal_encode.pkl"
scaler_pkl_file_name = "scaler.pkl"
model_file_name = "model.keras"

## Encoders and scalers
label_encoder=LabelEncoder()
onehotencoder=OneHotEncoder()
scaler = StandardScaler()


def read_data(file_to_train):
    data=pd.read_csv(file_to_train)
    return data

def clean_csv(data,fields,axis):
    return data.drop(fields,axis=axis)

def categorical_encode(data,fields):
    for field in fields:
        data[field]=label_encoder.fit_transform(data[field])
    return data

def ordinal_encode(data,fields):
    return onehotencoder.fit_transform(data[fields]).toarray()

def preprocess_data(data):
    ## Clean unwanted columns
    data=clean_csv(data,fields=unwanted_columns,axis=1)
    log(data.head())

    ## Encode categorical fields
    data=categorical_encode(data,categorical_fields)
    log(data.head())

    ## Encode ordinal fields
    one_hot_encoded_fields = ordinal_encode(data,ordinal_fields)
    log(one_hot_encoded_fields)

    ## encoded columns
    encoded_ordinal_columns= onehotencoder.get_feature_names_out(ordinal_fields)
    log(encoded_ordinal_columns)
    
    ## encoded dataframes
    encoded_df = pd.DataFrame(data=one_hot_encoded_fields,columns=encoded_ordinal_columns)
    log(encoded_df.head())

    ## Drop the original ordinal fields
    data = data.drop(ordinal_fields,axis=1)
    log(data.head())

    ## concat the encoded ordinal fields
    data = pd.concat([data,encoded_df],axis=1)
    log(data.head())
    
    return data

def log(text):
    print("")
    print(text)
    print("")


def save_pkl(encoder,filename):
    with open(filename,'wb') as file:
        pickle.dump(encoder,file)

def train():
    ## Load the train excel file
    data=read_data(file_to_train)
    log(data.head())

    ## clean and encode the data
    data= preprocess_data(data)

    ## save the encoders
    save_pkl(label_encoder,label_pkl_file_name)
    save_pkl(onehotencoder,ordinal_pkl_file_name)

    ## target and feature definition
    features_x = data.drop(target_field,axis=1) ## Features
    log(features_x.shape)
    
    target_y = data[target_field] ## Target
    log(target_y.shape)

    ## Split the data
    x_train,x_test,y_train,y_test = train_test_split(features_x,target_y,test_size=0.2,random_state=42)
    log(x_train.shape)

    ## Scale the data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    log(x_train)

    ## Save the scaler
    save_pkl(scaler,scaler_pkl_file_name)

    ## Build the model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))  ## Input layer
    model.add(Dense(64, activation='relu'))  ## hidden layer 1_shape=(x_train.shape[1],))) ## hidden layer 1
    model.add(Dense(32, activation='relu'))  ## hidden layer 2
    model.add(Dense(1, activation='sigmoid'))  ## output layer

    opt = tensorflow.keras.optimizers.Adam(learning_rate=0.01)
    loss = tensorflow.keras.losses.BinaryCrossentropy()

    ## Compile the model
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    ## Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    ## train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, callbacks=[early_stopping, tensorboard])

    ## Save the model
    model.save(model_file_name)




if __name__ == "__main__":
    train()