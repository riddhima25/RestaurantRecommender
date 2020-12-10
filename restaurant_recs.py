import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.recommendation import ALS
import time

bucket = "rest-recs-bucket"    # Replace with own bucket name
business_json_path = 'gs://{}/input/yelp_academic_dataset_business.json'.format(bucket)
user_json_path = 'gs://{}/input/yelp_academic_dataset_user.json'.format(bucket)
review_json_path = 'gs://{}/input/yelp_academic_dataset_review.json'.format(bucket)

def readfile(datatype):
    start = time.time()   
    print("{0} Start-time=\t{1}".format(datatype[0], time.ctime(start)))
    DF = spark.read.json(datatype[1])
    end = time.time()
    print("{0} End-time=\t{1}".format(datatype[0], time.ctime(end)))
    print("Time elapsed=\t{0} s \n".format(end-start))
    return DF

def flatten_df(nested_df):
    flat_cols = [c[0] for c in nested_df.dtypes if c[1][:6] != 'struct']
    nested_cols = [c[0] for c in nested_df.dtypes if c[1][:6] == 'struct']

    flat_df = nested_df.select(flat_cols +
                               [f.col(nc+'.'+c).alias(nc+'_'+c)
                                for nc in nested_cols
                                for c in nested_df.select(nc+'.*').columns])
    return flat_df

def preprocess_bus_data(df):
    # TODO: Need to handle categories field + assemble into vector
    df = flatten_df(df)
    cat_cols = list(set(df.columns) - set(['is_open', 'latitude', 'longitude', 'review_count', 'stars']))
    ind_cols = [col+'_ind' for col in cat_cols]
    indexers = [
        StringIndexer(inputCol=col, outputCol=new_col)
        for col, new_col in zip(cat_cols, ind_cols)
    ]
    pipeline = Pipeline(stages=indexers)
    processed_df = pipeline.fit(df).transform(df)
    return processed_df

def preprocess_review_data(df): 
    udf = UserDefinedFunction(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')), DoubleType())
    df = df.withColumn('date_num', udf(df.date))
    df = df.drop('date')
    cat_cols = ['business_id', 'review_id', 'user_id']
    index_cols = [col+'_ind' for col in cat_cols]
    indexers = [
        StringIndexer(inputCol=col, outputCol=new_col)
        for col, new_col in zip(cat_cols, index_cols)
    ]
    
    # Causing this to take a long time + not sure if needed:
    # vector_cols = [col+'_vector' for col in cat_cols]
    # encoder = [OneHotEncoderEstimator(inputCols=index_cols, outputCols=vector_cols)]
    
    # not processing review text for now
    feat_cols = ['business_id_ind', 'cool', 'date_num', 'funny', 'stars', 'useful', 'user_id_ind']
    assembler = VectorAssembler(inputCols=feat_cols,outputCol="vector")
    
    pipeline = Pipeline(stages=indexers+[assembler])
    model = pipeline.fit(df)
    processed_df = model.transform(df)
    # split = processed_df.randomSplit([0.7,0.3], seed=100)
    return processed_df
    
def preprocess_user_data(df):
    # TODO: Need to figure out how we want to represent friends attribute + assemble into vector
    udf = UserDefinedFunction(lambda x: len(x.split(',')) if x != '' else 0, IntegerType())
    df = df.withColumn('num_years_elite', udf(df.elite))
    df = df.drop('elite')
    udf = UserDefinedFunction(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')), DoubleType())
    df = df.withColumn('yelping_since_num', udf(df.yelping_since))
    df = df.drop('yelping_since')
    return df

if __name__ == '__main__':
    spark = SparkSession.builder.appName('Yelp Data').getOrCreate()  

    # Reading data into dataframes  
    business = ['Business Data', business_json_path]
    user = ['User Data', user_json_path]
    review = ['Review Data', review_json_path]

    busDF = preprocess_bus_data(readfile(business))
    userDF = preprocess_user_data(readfile(user))
    reviewDF = preprocess_review_data(readfile(review))

    # TODO: Need to figure out why this is causing heap error
    # Attempt at collaborative filtering (causes heap error):
    # als = ALS(maxIter=5, regParam=0.01, userCol="user_id_ind", itemCol="business_id_ind", ratingCol="stars",
    #       coldStartStrategy="drop")
    # model = als.fit(reviewDF)
    # userRecs = model.recommendForAllUsers(10)
    # #can also do: itemRecs = model.recommendForAllItems(10) to get top user recs for each movie    
    spark.stop()

