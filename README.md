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
