# AIS2025Intern_Akash_Gawade
# Task 1
1) Dictionary:
  - Creating a Dictionary and Running basic Algorithms like Finding length, Data types, Adding different data types Like Integer, Float, boolean, String, etc.
2) LIst :
  - Creating list and running a basic algorithm. Modifying elements
  	Using .append() and .insert() to add the elements
  	Removing Elements using .remove(), pop(), clear()
  	List is a mutable dataset so we can be done sorting, reverse, extend, copy
3) Operators :
  - Operators in Python
	     1. Arithmetic ( + , - , / , *, ** )
	     2. Comparative ( == , != , < , > ,<= ,>=)
	     3. Logical ( and , or )
	     4.Assignment ( -=, += , *=)
	     5. Identy ( is(==) , is not(!=))
	     6. Membership ( in , not in)
4) Common Errors in Python:
  - Discussion some basic pthon errors:
  		ZeroDivisionError
		Name Error
		Value Error
		Index error
		Type error
		Identeation Error
		Key error
5) Set :
- Set is one of 4 built-in data types in Python used to store collections of data, the other 3 are List, Tuple, and Dictionary, all with different qualities and usage.
- holds some discussion on data types and Fuctions related to  the Set
6) Tuple :
- Tuple is a collection which is ordered and unchangeable. Allows duplicate members.
  
7) Numbers :
- There mainly 3 types in Python

	1.Integers
	    Def: Whole Number without a fractional or decimal point. They can be Positive or may be Negative or zero. Ex. 77, -58,0, etc
	    
	2.Floats
	    Def: Numbers that contain a decimal point or are expressed in scintific notation 
	    Ex. 2.3,00.15.-0.2545
	3. Complex Number
	    Def : Number in the form of a+bj where a is real part and the b is imaginary part


# Task 2:
1) For Loop:
- for loop is used for do iterstion overe a specific sequence

2) If else :
- if else is condiation al satetment that allows user to set the a condition for iteartion.
This contain two part of code 1 st one is 'if' and 2nd is 'else'.
If the first condition Is False then iteration forwarded two else condition.

3) Exapmle :
- Given a list, write a Python code to swap first and last element of the list.

write code count lenght of string

Write a Python program to get the sum of a only non-negative integer. ex, [1,4,-5,-20,10] ans is 15
write code of factorial , ex.ans 6 (321)


4) While loop :
- In While loop, iteration done upto unknown limitaion with condition for stoping the itearation. We dont perfectly where the iteration can be done.
- #1)odd-even using while loop 
- 2) using while
	* 
	* *
	* * *
	* * * *
	* * * * *
- 3) creat list 1-20 numbers list using while loop=> [1,2,3... 20]
- 4)  creat list 20-1 (revers order) using while loop=> [20,19...1]
- 5) try with one any eg. break, contnue , pass control statement
# Task 3
1) Def:
  * def is helps to make user defined function . Containning the code of block with name of function
* Some Example
# Task 4
1) Numpy
* NumPy library provides the mathematical operation at once on the array for computation, it is Used for linear algebra, random number generation , and many more. Numpy provides support for large multi-dimensional array and matrices, which are important for scientific computation. It is faster than did the operation on the Python list. It also works with libraries such as Pandas, Matplotlib , Scipy, etc.
* Ceating arrays
* Arithmetic Operations
* Indexing and Slicing
* Shape and reshape
* Broadcasting
* Array manipulation
* Mthematical Functions
* Statistical Fun
* Linear Algebra fun
* Random Number Genration
* Sampling and Resampling

# Task 5
Pandas :
	* Pandas is the powerfull tool in the python . Pandas provide different data structures like series and dataframe. series is the 1D array and datafram is the 2D data table like Excel Spreadshit. We can spliit the data by indexing using integer , We can split the specific column of the data for special operations
	Pandas done manipulation of data
	Cleaning of data
	Data analysis
	We can explore the data
 * Data Structure
 * Series
 * DataFrame
 * Data Preprocessing
 * Oprations on Columns
 * Filtering
 * Aggregation and  Grouping
 * Merging and Joining
 * File input Output
 * Basic Plots
# Task 6
Matplot:
* In python matplot library used for basic Visualiztion
Marplotlib itegrate with other libraries in the python such as Numpy , Pandas ,Scipy.
Matplot have many varieties in the graphs auch as Line, Scatter Histogram ,Bar chart, pie chart , joit graph and many more
Matplotlib was originally written by John D. Hunter. Since then it has had an active development community and is distributed under a BSD-style license.
  * we discussed some types og Graph
	  * Line Plot(With marker
 	  *  Scatter Pliot
   	  * Histogram (with grid, color )
    	  * Bar Plot
    		* Vertical
        	* Horizonatl
          * Box Plot
          * Pie chart (with modification)
          *  Subplots (Vertical and Horizontal)
Seaborn :
*Statistically more important¶
For Advanced Visualization we use the Seaborn library
Seaborn esxcel at creting plots that incorporate the statistical elements, like confidence interal
Variuos Plot like ,matrix , relational,Distributional,joint

  	  *  Heatmap
          * Pairplots (using built in data set 'iris and self genrated data)
          * Voilin plot using 'tips' data
          * Distribution plots (KDE plots)
          * Joit Plots ( SelfGenrated data and Random data )

# Task 7
Titanic Data set:
 * We did some basic informatic process's
   	* Loading the Dataset
   	* Checking the Statistical Parameter using .describe()
   	* by using the .info() we able to see the data type , column name  and which column having missing values.
   	* By using the .shape() we obatian the size of data
* Data Preprocessing
	* Data Proprocessing include Finding Missing Values
	* Handling Missing values
* Label Encoding
  	* Label Encoding is used for convert the Categorical data into Numerical data or formate.



# Task 8
Titanic DataSet:
* Pre-Processing
  	* Handling Missing values
* Finding Qurtiles
  	* Q1 and Q3
  	* IQR
* Using Tukeys Method for Outlier Detection
  	* by using calculated IQR
* Box Plot And Whisker Plot
	* By using Seaborn and Matpllot Libraries
* Z-Score
  	* By using Scipy libarary
* Standardization
  	* Transform data into form having mean 0 and standard deviation 1.
* MinMaxScalers
* Principal Component Analysis
  	* Reduce the data into minimum dimension

# Task 9
Iris DataSet
* ### for Handling data we used pandas and numpy
* ### To machine learning algorithm we used SciKit Learning Library
* Loading Dataset using SciKit learning package
  	* Convert into pandas dataframe
* Separate Dependent Variable and Independent Variable
* Splitting the dataset into Training and Testing sub-dataset.
 	* 80% of data used for training the model
 	* 20% of data used to test the model
* Model Building
  	* GaussianNB
  	* BernoulliNB
  	* MultinomialNB 
* Training Models
* Prediction
* Accuracy Score
  	* Our Data contain Continuous data values, So for continuous data, Our Gaussian Naive Bayes Models give the Best prediction, Meanwhile Bernoulli and Multinomial are not more suitable than the Gaussian model.
	* Accuracy Score for each mode
   		* GaussianNB Accuracy= 1.0
       		* BernouliNB= 0.3
           	* MultinomialNB Accuracy Score= 0.9


# Task 10:
Decision Tree
### Theory:
* A supervised ML algorithm for classification/regression.
* A Decision Tree in statistics is a tree like structure having braches and nodes.
* A tree have Main branch is Root Node, branches , and leaf node.
* Every branch contain a spliying criterion.
* Every Leaf node have class label like (Yes \ No), (0,1)

## Entropy and GINI
 Both are pliting criteriain
 * Entropy :
  		* Measure the Disorder
  		* Entropy is Logarithmic Form
    		* Slow to compute 
      		* ![image](https://github.com/user-attachments/assets/d2c3e0a9-1b15-4ec7-941a-9ca7c3d8a80c)

* GINI:
 		* Measures the probability of misclassification
   		* Easy to Apply
     		* Simple to Compute
       		* ![image](https://github.com/user-attachments/assets/9637ef57-ec53-421c-a012-d708bad8d928)
* Imorted from Sklearn.tree package
* having same algorithm upto criterion
  	* # Train Decision Tree using Gini criterion
         dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
  	  
        dt_gini.fit(X_train, y_train)
  	  
        y_pred_gini = dt_gini.predict(X_test)
  	* # Train Decision Tree using Entropy criterion
	   dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

	    dt_entropy.fit(X_train, y_train)
  	  
	     y_pred_entropy = dt_entropy.predict(X_test)





## Projects

# Machine Learning
Problem Statement
Business Problem
"Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers."

 Loan prediction is a very common real-life problem that every retail bank faces in their lending operations. If the loan approval process is automated, it can save a lot of man hours and improve the speed of service to the customers. The increase in customer satisfaction and savings in operational costs are significant. However, the benefits can only be reaped if the bank has a robust model to accurately predict which customer's loan it should approve and which to reject, in order to minimize the risk of loan default.



Translate Business Problem into Data Analysis / Machine Learning problem:
This is a classification problem where we have to predict whether a loan will be approved or not. Specifically, it is a binary classification problem where we have to predict either one of the two classes given i.e. approved (Y) or not approved (N). Another way to frame the problem is to predict whether the loan will likely to default or not, if it is likely to default, then the loan would not be approved, and vice versa. The dependent variable or target variable is the Loan_Status, while the rest are independent variable or features. We need to develop a model using the features to predict the target variable.

 Machine Learning Work 
 
# Data Collection
			On the first day We collected the data From the Kaggle and Hackthon Web-pages. The Chosen dataset  must required in sufficient amount. Our data present in two CSV file. The Train.csv contain training dataset and other one test.csv Contain testing dataset. Whole dataset splitted into 80:20 proportion.
	
# tour on dataset
			Our dataset contain 13 variable and 613 observations in Training dataset. Our target variable is Loan_Status. Test dataset contain 12 variable and 366 observation for testing the model.
	


 Data Preprocessing :
	Data preprocessing is a data mining technique that involves transforming raw data into an understantable format. Real-World data is often incomplete, inconsistent, and/or lacking in certain behaviour, and is likely to contain many errors. Data Preprocessing is a method if resolving such issues.

Outlier Treatment
As we saw earlier in univariate analysis, LoanAmount contains outliers so we have to treat them as the presence of outlier affects the distribution of the data. Having outlier in the dataset often has a significant effect on the mean and standard deviation. Hence affecting the distribution . We must take steps to remove outliers from out dataset.
Due to these outlier  of the data in the Loan Amount is at the left and the right tail is longer. This called right skewness or positive skewness. One way take the log transformation, it does note affect the smaller values much, but reduces the larger values. So, we get a distribution similar to normal distribution.

Missing Value 
After exploring all other variables in our data, we can now impute the missing values and treat the outliers because missing data 
There is missing value in Gender, Marital Status, Dependents, Credit History and self employment, loan amount,  loan amount train. We will treat the missing values in all the features one by one.
We can consider this methods to fill the missinhg values:
    - For numerical variable : imputation using mean and median
    - For categorical variable : imputation using mode
There are very less missing values in Gender , Married, Dependent, Credit_History and Self_Employed features so we can fill them using the mode of that feature, If an independent variable in our dataset has huge amount of missing data e.g. 80% missing values in it, then we would drop the variable from the dataset


Model Development and Evaluation
There are four sub-sections in this stage:
- Evaluation Martics for Classification Problem
- Model Building Part - I
- Feature Engineering
- Model Building Part- II
-Evaluation Metrics for Classification Problems

Evaluation Metrics for Classification Problem
The process of model building is not complete without evaluation of model’s performance. Suppose we have the predictions from the models, how we can decide whether the prediction are accurate? We can plot the results and compare them with the actual values. i.e. calculate the distance between the predictions and actual values. Lesser this distance more accurate will be the predictions. Since this is a classification problem, we can evaluate our models using any one of the following evaluation metrics:
Accuracy : Let us understand it using the confusion matrix which is a tabular representation of Actual vs Predicational values. This is how a confusion matric look like :
True Positive - Targets which are actually true(Y) and we have predicted them true(Y)
True Negative - Targets which are actually false(N) and we have predicted them false(N)
False Positive - Targets which are actually false(N) but we have predicted them true(T)
False Negative - Targets which are actually true(T) but we have predicted them false(N)
Using this values, we can calculate the  accuracy of the model. The accuracy is Given by :

Accuracy = (TP+TN)/(TN+FN+TP+FP))

Precision : It is measure od correctness achieved in true prediction i.e. of observation labelled as true, how many are actually labelled true.
Precision= TP/(TP+FP)

Recall(Sensitivity) : It is measure of actual observations which are predicted correctly i.e. how many observations of true class are labled correctly. It is also known as 'Sensitivity'. eg. Proportion of patients with a disease who test positive.
Recall = TP/(TP+FN)

Specificity : it is measure of how many obseravtions of False Class are labled correctly. e.g. Proportion of patient without the disease who test
Specificity = TN/(TN+FP)


ROC curve

Receiver Operating Characteristic(ROC) summarizes the model’s performance by evaluating the trade offs between true positive rate (Sensitivity) and false positive rate (1- Specificity).
- The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction power of the model.
- The area of this curve measures the ability of the model to correctly classify true positives and true negatives. We want our model to predict the true classes as true and false classes as false.
- So it can be said that we want the true positive rate to be 1. But we are not concerned with the true positive rate only but the false positive rate too. For example in our problem, we are not only concerned about predicting the Y classes as Y but we also want N classes to be predicted as N.
- We want to increase the area of the curve which will be maximum for class 2,3,4 and 5 in the above example.
- For class 1 when the false positive rate is 0.2, the true positive rate is around 0.6. But for class 2 the true positive rate is 1 at the same false positive rate. So, the AUC for class 2 will be much more as compared to the AUC for class 1. So, the model for class 2 will be better.
- The class 2,3,4 and 5 model will predict more accurately as compared to the class 0 and 1 model as the AUC is more for those classes.

Models for Classification:
-1) Logistic Regression
-2) Decision Tree Classifier
-3) Random Forest Classifier
-4) XGBoost Classifier

Data Spliting For Model Building:

- Using the SciKit learn library we can import the train_test_split model fro split the data into training and testing dataset
We also define the test size for proportion of data splitting like 80:20 or 70:30 .

Model Building : Part I
1)Logistic Regression
Let us make our first model to predict the target variable. We will start with Logistic Regression which is used for predicting binary outcome.

Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.  Logistic regression is an estimation of Logit function. Logit function is simply a log of odds in favour of the event. This function creates a s-shaped curve with the probability estimate, which is very similar to the required step wise function
Accuracy For Logistic Regression is 82%.

2)Decision Tree
Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.

Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable.
Accuracy For Decision Tree is 65%.

3)Random Forest
Random Forest is a tree based bootstrapping algorithm wherein a certain no. of weak learners (decision trees) are combined to make a powerful prediction model.
For every individual learner, a random sample of rows and a few randomly chosen variables are used to build a decision tree model.
Final prediction can be a function of all the predictions made by the individual learners.
In case of regression problem, the final prediction can be mean of all the predictions.
There are some parameters worth exploring with the SciKit-learn Random Forest Classifier:

•	n_estimators
•	max_features
n_estimators  usually bigger the forest the better, there is small chance of overfitting here. The more estimators you give it, the better it will do. We will use the default value of 10.

max depth of each tree (default none, leading to full tree) - reduction of the maximum depth helps fighting with overfitting. We will limit at 10.


4) GridSearchCV
We will try to improve the accuracy by tunning the hyperparameter for this model. We will use grid search to get the optimized values of hyperparameter . Grid Search is a way to select the best of as family of hyper parameters, parameterized by a grid of parameter.

We will use GridSearchCV in sklearn.model_selection for an exhaustive search over specified parameter values for an estimator. GridSearchCv implement a 'fit' and a 'score' method. It also implement 'predict', 'predict_proba' , 'decision_function', 'transform' and 'inverse_transform' if they are implanted in the estimator used. We will tune the max_depth and n_estimators parameters. max_depth decides the maximum depth of the tree and n_estimators decides the number of trees that will be used in random forest model.

5) Random Forest Classification with Grid Search CV
# Provide range for max_depth from 1 to 20 with an interval of 2 and from1 to 200 with an interval for n_estimators , by undertaking the Code:
paragrid={'max_depth':list(range(1,20,2)),'n_estimators':list(range(1,200,20))}
In this model , we got best_estimator_ value as 181. Let's next step was build the model using with optimized value
Accuracy is 80%.

6) XGBoost
XGBoost is a fast and eficient algorithm and has been used to by the so many data scientist. XGBoost works only with numeric variables and we have already replaced the categorical variables with numeric variables. Let’s have a look at the parameters that we are going to use in our model.
n_estimator: This specifies the number of trees for the model.
max_depth: We can specify maximum depth of a tree using this parameter.
Accuracy for XGBoost is 80%.


Conclusion For Machine Learning Project:
After trying and testing 4 different algorithms, the best accuracy is achieved by Logistic Regression (0.7847), followed by Random Forest (0.7904) and XGBoost (0.80), and Decision Tree performed the worst (0.7197). Compared to using default parameter values, Grid Search CV helped improved the model's mean validation accuracy by providing the optimized values for the model's hyperparameters. On the whole, a logistic regression classifier provides the best result in terms of accuracy for the given dataset, without any feature engineering needed. Because of its simplicity and the fact that it can be implemented relatively easy and quick, Logistic Regression is often a good baseline that data scientists can use to measure the performance of other more complex algorithms. In this case, however, a basic Logistic Regression has already outperformed other more complex algorithms like Random Forest and XGBoost, for the given dataset.


Model Deployment On Streamlit :
After the successfully running the model code with best accuracy  has selected for making the deployment. Our Logistic Regression Model  gives Highest Accuracy Score. For making the app we used "spyder" for python code writing and Streamlit for Deployment. We used Streamlit , Pickle libraries for making the code easier.
Streamlit:
    BYy using the Streamlit we can create web page for the deployment and Desing of the page. We added the Buttons for prediction with Logistic Regression and Random Forest. When we run the deployment algorithm in the terminal, a simple web page open in the Browser. We can set the Title , header , footer, colour and so many features. 
We Create the inputs for single instance or observation with respect to the training Features. We can put the value of the Features on the web app and after this  we can Predict the Target variable. In this Project Work We have 11 Independent Variables and 12th is target variable. prediction made on this values , for verification we can try to enter the values from the training data from the csv file. High Accuracy means chance of correct class prediction is High.

Pickle:
By using the pickle library we can save model in the file like as “combineM2.pkl”. Its helps to make the Streamlit code easier to enter the model in the application deployment.



Statistics Insight the Project Work:
The Statistical Algorithm like Data Collection, Data preprocessing, Data splitting, Missing Value Imputation, Classification algorithm, Outlier Treatment, Evaluation Matrix, Descriptive Statistics, and so many are used. When Data Contain missing values then, is can be made a bias for the Hypothesis Testing and class prediction. If the Continues class values contain the missing value with presence of outlier then, For fill the missing choosing the Median than mode is effective. Filling the missing value with mean it can be decrease the accuracy, because the outlier value force to move toward the Highest or Lowest Value. Then We can use the IQR method and Min-Max method to remove it. When outlier are presence in which we used the median. For the categorical value we used the Mode.
	For Data splitting we used 80:20 Proportion.
