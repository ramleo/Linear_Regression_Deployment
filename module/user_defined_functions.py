
# Creating function for importing and splitting the dataset

def impt_split_data(data):
    '''This function splits the data and outputs copies of
       X_train, X_test, y_train and y_test'''

    # Importing dataset
    df = pd.read_csv(data)

    # Dropping unnecessary cloumns
    df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)

    # Dividing bikes dataset in to X, y dataset
    X = df.drop('cnt', axis=1)
    y = df['cnt']

    # Splitting the X and y datasets into Train sets and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


def prep_data(X_data):

    '''This function converts the discrete variables into categorical variable
       and outputs copy of the dataset'''
    # Creating copies of train data
    X_data = X_data.copy()

    # Converting categorical variables to object as it is a requirement
    X_data['season'] = X_data[['season']].apply(lambda x: x.map({1: 'spring', 2: 'summer',
                                                                 3: 'fall', 4: 'winter'}))

    X_data['weathersit'] = X_data[['weathersit']].apply(lambda x: x.map({1: 'Clear',
                                                                         2: 'Mist_Cloudy',
                                                                         3: 'LightRain_LightSnow'}))

    X_data['mnth'] = X_data[['mnth']].apply(lambda x: x.map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                                                             5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                                                             9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}))

    X_data['weekday'] = X_data[['weekday']].apply(lambda x: x.map({0: 'Sun', 1: 'Mon', 2: 'Tue',
                                                                   3: 'Wed', 4: 'Thur', 5: 'Fri',
                                                                   6: 'Sat'}))

    return X_data.copy()


def onehot_encode(X_data=None):
    '''This function dummy encoding of the dataset using sklearn one hot encoder'''
    X_data = prep_data(X_train)

    ohe = OneHotEncoder(sparse=False)

    tmp = ohe.fit_transform(X_data[['season', 'weathersit', 'mnth', 'weekday']])

    return ohe, X_data, tmp


def rfe_select(X_data=None, y_data=None, onehotencoder=None, ohe_array=None, features_to_select=12):
    '''This function performs sklearn RFE to select important features which are then provided to the model'''
    enc_feat = []
    for f in onehotencoder.categories_:
        enc_feat.extend(list(f))

    X_data.reset_index(inplace=True)

    tmp = pd.DataFrame(ohe_array, columns=enc_feat)

    X_data = pd.concat([X_data, tmp], axis=1)

    X_data.drop(['index', 'season', 'weathersit', 'mnth', 'weekday'], axis=1, inplace=True)

    rfe = RFE(LinearRegression(), n_features_to_select=features_to_select)

    rfe.fit(X_data, y_train)

    X_data = X_data[X_data.columns[rfe.support_]]

    sel_cols = X_data.columns

    return X_data, sel_cols, enc_feat


def prep_fit_traindata(X_data=None, y_data=None):
    '''This function uses rfe function, minmax function and fit transforms the dataset, also it
    trains the dataset using sklearn linearRegression function'''
    ohe, X_data, tmp = onehot_encode(X_data=X_data)

    # OneHot encoding and feature selection using RFE
    X_data, sel_cols, enc_feat = rfe_select(X_data=X_data, y_data=y_train, onehotencoder=ohe, ohe_array=tmp)

    # Initiating pipeline
    minmax = MinMaxScaler()

    # Transforming train data
    X_data = minmax.fit_transform(X_data)

    # Fitting Linear Regression model
    lr = LinearRegression()
    model = lr.fit(X_data, y_data)

    return ohe, minmax, model, sel_cols, enc_feat
