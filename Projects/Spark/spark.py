# solutions.py

import pyspark
from pyspark.sql import SparkSession
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MCE



# --------------------- Resilient Distributed Datasets --------------------- #

### Problem 1
def word_count(filename='huck_finn.txt'):
    """
    A function that counts the number of occurrences unique occurrences of each
    word. Sorts the words by count in descending order.
    Parameters:
        filename (str): filename or path to a text file
    Returns:
        word_counts (list): list of (word, count) pairs for the 20 most used words
    """ 
    # create SparkSession
    spark = SparkSession\
        .builder\
        .appName("Word Count")\
        .getOrCreate()
    
    # read text file
    rdd = spark.sparkContext.textFile(filename)

    # count words
    word_counts = rdd.flatMap(lambda line: line.split()) \
        .map(lambda word: (word, 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: x[1], ascending=False) \
        .take(20)
    
    # stop SparkSession
    spark.stop()

    # return word counts
    return word_counts

    
    
    
    
### Problem 2
def monte_carlo(n=10**5, parts=6):
    """
    Runs a Monte Carlo simulation to estimate the value of pi.
    Parameters:
        n (int): number of sample points per partition
        parts (int): number of partitions
    Returns:
        pi_est (float): estimated value of pi
    """
    # create SparkSession
    spark = SparkSession\
        .builder\
        .appName("Monte Carlo")\
        .getOrCreate()
    
    # calculate pi
    rdd = spark.sparkContext.parallelize(range(parts)) \
        .map(lambda x: np.random.rand(n, 2)) \
        .map(lambda x: np.sum(np.sum(x**2, axis=1) < 1)) \
        .reduce(lambda a, b: a + b)
    pi_est = 4 * rdd / (n * parts)

    # stop SparkSession
    spark.stop()

    # return pi
    return pi_est

# ------------------------------- DataFrames ------------------------------- #

### Problem 3
def titanic_df(filename='titanic.csv'):
    """
    Calculates some statistics from the titanic data.
    
    Returns: the number of women on-board, the number of men on-board,
             the survival rate of women, 
             and the survival rate of men in that order.
    """
    # create SparkSession
    spark = SparkSession\
        .builder\
        .appName("Titanic Df")\
        .getOrCreate()
    
    # create schema
    schema = ('survived INT, pclass INT, name STRING, sex STRING, '
            'age FLOAT, sibsp INT, parch INT, fare FLOAT'
            )

    # read csv file
    titanic = spark.read.csv(filename,schema=schema)
    
    # create temporary view
    titanic.createOrReplaceTempView("titanic")

    # calculate statistics
    total_count = spark.sql(
        "SELECT sex, COUNT(*) AS count\
        FROM titanic\
        GROUP BY sex"
    ).collect()

    survival_rate = spark.sql(
        "SELECT sex, AVG(survived) as survival_rate\
        FROM titanic\
        GROUP BY sex"
    ).collect()

    # stop SparkSession
    spark.stop()

    # return statistics
    return (total_count[0][1], total_count[1][1], survival_rate[0][1], survival_rate[1][1])



### Problem 4
def crime_and_income(crimefile='london_crime_by_lsoa.csv',
                     incomefile='london_income_by_borough.csv', major_cat='Robbery'):
    """
    Explores crime by borough and income for the specified major_cat
    Parameters:
        crimefile (str): path to csv file containing crime dataset
        incomefile (str): path to csv file containing income dataset
        major_cat (str): major or general crime category to analyze
    returns:
        (ndarray): borough names sorted by percent months with crime, descending
    """
    # Create SparkSession
    spark = SparkSession.builder.appName("Crime and Income Analysis").getOrCreate()
    
    # Load Data
    crime_data = spark.read.csv(crimefile, header=True, inferSchema=True)
    income_data = spark.read.csv(incomefile, header=True, inferSchema=True)

    # Create temporary views
    crime_data.createOrReplaceTempView("crime")
    income_data.createOrReplaceTempView("income")

    # Create joined table
    df = spark.sql(
        f"SELECT crime.borough, `income`.`median-08-16`, SUM(crime.value) AS major_cat_total_crime\
        FROM crime\
        JOIN income\
        ON crime.borough = income.borough\
        WHERE major_category = '{major_cat}'\
        GROUP BY crime.borough, `income`.`median-08-16`\
        ORDER BY major_cat_total_crime DESC"
    )

    numpy_array = np.array(df.collect())

    # Stop SparkSession
    spark.stop()
# Function to format tick labels
    def custom_formatter(x, pos):
        return f'{x/10**3:.0f}K' if x >= 1000 else f'{x:.0f}'

    formatter = FuncFormatter(custom_formatter)

    # Create scatter plot of data, rounded to the hundreds
    plt.scatter(numpy_array[:, 1], numpy_array[:, 2])
    plt.xlabel('Median Income')
    plt.ylabel('Total Crime')
    plt.title('Total Crime vs Median Income')

    # Set custom formatters for axes
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()

    # return array
    return numpy_array[:, 0]



### Problem 5
def titanic_classifier(filename='titanic.csv'):
    """
    Implements a classifier model to predict who survived the Titanic.
    Parameters:
        filename (str): path to the dataset
    Returns:
        metrics (tuple): a tuple of metrics gauging the performance of the model
            ('accuracy', 'weightedRecall', 'weightedPrecision')
    """

    # create SparkSession
    spark = SparkSession\
        .builder\
        .appName("Titanic Classifier")\
        .getOrCreate()
    
    # create schema
    schema = ('survived INT, pclass INT, name STRING, sex STRING, '
            'age FLOAT, sibsp INT, parch INT, fare FLOAT'
            )

    # read csv file
    titanic = spark.read.csv(filename,schema=schema)

    # convert the 'sex' column to a numeric column
    sex_binary = StringIndexer(inputCol='sex', outputCol='sex_binary')

    onehot = OneHotEncoder(inputCols=['pclass'], outputCols=['pclass_onehot'])

    # create a feature vector
    features = ['sex_binary', 'pclass_onehot', 'age', 'sibsp', 'parch', 'fare']
    feature_vector = VectorAssembler(inputCols=features, outputCol='features')

    # create a pipeline
    pipeline = Pipeline(stages=[sex_binary, onehot, feature_vector])
    titanic = pipeline.fit(titanic).transform(titanic)

    # split the data into training and test sets
    train, test = titanic.randomSplit([0.75, 0.25], seed=11)

    # create a random forest classifier
    rf = RandomForestClassifier(featuresCol='features', labelCol='survived', seed=42)

    # create a parameter grid
    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 100]).build()

    # create an evaluator
    evaluator = MCE(labelCol='survived', metricName='accuracy')

    # create a cross validator
    tvs = TrainValidationSplit(estimator=rf,
                                 estimatorParamMaps=paramGrid,
                                 evaluator=evaluator,
                                 trainRatio=0.8,
                                 seed=42)
    
    # fit the model
    clf = tvs.fit(train)

    # make predictions
    predictions = clf.transform(test)

    # calculate metrics
    metrics = (evaluator.evaluate(predictions),
                MCE(labelCol='survived', metricName='weightedRecall').evaluate(predictions),
                MCE(labelCol='survived', metricName='weightedPrecision').evaluate(predictions))
    

    # stop SparkSession
    spark.stop()

    # return metrics
    return metrics
 

if __name__ == "__main__":
    print(word_count())
    print(monte_carlo())
    print(titanic_df())
    print(crime_and_income())
    print(titanic_classifier())