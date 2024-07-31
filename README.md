# Price Prediction of Used Cars using ML Part 2

The objectives of this project are :-
1. Learn how to one hot encode categorical data.
2. To check for prediction improvement using the categorical feature (Car_Name).

In part 1, while predicting the **Selling_Price** of the used car, we have dropped the column **Car_Name** from the feature set. The **Car_Name** represents the car model, which I think plays a crucial role when predicting the price of that car.

You can refer to the first part of this project [here](https://github.com/sreedevnair/Price-prediction-of-used-cars-using-basic-ML). (It's not a continuation of that project, just kind of an another iteration)

<br>

## 1. Learn how to one hot encode categorical data.
One-hot encoding : It is a technique that we use to represent categorical variables as numerical values in a machine learning model.

It basically creates new columns indicating the presence (or absence) of each possible value in the original data.

Even though it's a small database with only 301 entry, there's a total of 98 unique **Car_Name**. One-hot encoding is not much effecient if the cateogrical feature has large number of unique values (usually less than 15 is considered good).

To get the unique values, we have used the function `df['Car_Name'].unique()`. It will return an array of all the possible values of **Car_Name**.

So the first thing to do is to drop all the car details whose **Car_Name** appears less than 2 times. It will help the model to find better patterns between the **Car_Name** and **Selling_Price**.  To do this, we used `.groupby()` fucntion along with lambda function and then finally saved the data in a new DataFrame `new_df = df.groupby('Car_Name').filter(lambda x: len(x) >= 3)`.

Now, the new DataFrame contains a total of 34 unique values.

There are 2 ways to perform one hot encoding :-
1. Using pandas `.get_dummies()`
2. Using sklearn `OneHotEncoder()` Class

### 1.1 Using pandas `.get_dummies()`
It is probably the simplest way to perform One-hot encoding. In this, we use a Pandas' function `.get_dummies()` and inside the brackets, we pass the column from dataset which we want to encode along with the `dtype`. 

Then, it will create a new DataFrame in which all the unique possible of **Car_Name** will be listed as columns and 1 will be used to represent under which **Car_Name** a particular rows falls into.

We have to specity `dtype` else if will give a DataFrame with *True* and *False* values instead of 1 and 0.

Once we have done that, we will simple concatenate the this encoded DataFrame with our previous DataFrame using `pd.concat([list of DFs], axis=1)`.

Now, we can drop the column **Car_Name** and also another one encoded column. Its because if every one hot encoding entries are zeroes in a row, then that will also tell the model that it falls into a different category.

(For this logic, kindly ignore the fact that I have already dropped 64 unique values of **Car_Name** 👉🏻👈🏻)

This is also known as ***dummy variable trap***.

### 1.2 Using sklearn `OneHotEncoder()` Class
When have to first import the class `OneHotEncoder()` ***sklearn.preprocessing***. After that, we initialize the class object `OneHotEncoder(sparse_output=False).set_output(transform='pandas')`.

Here, we need the output to be in the form of Pandas and not arrays. Hence we have used the function `.set_output(transform='pandas')`.

After that, we use the function `.fit_transform()` and inside the brackets, we pass the column from dataset which we want to encode (just like we did using Pandas' function). 

It will create a one-hot encoded DataFrame. Then we simple concatenate this DataFrame with our previous DataFrame and drop the column **Car_Name** and also another one encoded column from the new DataFrame.

<br>

## 2. To check for prediction improvement using the categorical feature.
Here, we will be training our model angain through Lasso Regression.

We train and predict the model in the same way we train Decision Tree model or Random Forest model.

For evaluating the model, we have used the error metrics known as R squared error. The value of R-square lies between 0 to 1.

Closer the R score error value is to 1, the better our model did at predicting the target value.

Finally, our model's R square error using Lasso Regression Model is 0.91, which is almost twice the score from our previous prediction, 0.46.

<br>

## Conclusion 
After encoding the column **Car_Name** through one-hot encoding, our model performed way better at predicting the target value.
