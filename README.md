# profit_prophet_data_processing
Preprocessing data and creating stock market predictions with Tensorflow Keras

# Directory structure

- All the csv raw data are stored in tha `.\data` folder
- The data needs to be split into training and testing sets, I split the dataset with
a `test_size=0.2` with scikit learn library, and turned shuffle off

- training data is stored in `.\data_train`
- testing data is stored in `.\data_test`
- until now all stocks are stored separately
- now the data needed to be aggregated to one file to be able to feed the neural network all at once:
in `.\data_test\aggregated` and `.\data_train\aggregated` I aggregated all the data to one file while converting
all data to numbers: datetime relative in days to `1970-01-01`, adding a ticker field to be able to separate stocks
and adding an `Is Stock` boolean field (0 or 1)
- after aggregating all the data to files, for further preprocessing I needed to scale data to eliminate biases from different sizes of values
- 


