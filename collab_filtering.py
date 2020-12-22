import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.sql.functions import UserDefinedFunction, array
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.recommendation import ALS
import time
from pyspark.sql.functions import col, unix_timestamp

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
    
def preprocess_user_data(df):
    drop_list = ['compliment_cool', 'compliment_cute', 'compliment_hot', 'compliment_list', 'compliment_more', 
                    'compliment_note', 'compliment_photos', 'compliment_plain', 'compliment_profile', 
                     'compliment_writer', 'compliment_funny', 'friends', 'name']
    users = df.select([c for c in df.columns if c not in drop_list])

    # get all features to long/double
    indexer = StringIndexer(inputCol='user_id', outputCol='user_id_int')
    userIndexModel = indexer.fit(users)
    users = userIndexModel.transform(users)

    users = users.withColumn("yelping_since_date", unix_timestamp(col("yelping_since")))
    users = users.drop('yelping_since')

    udf = UserDefinedFunction(lambda x: len(x.split(',')) if x != '' else 0, IntegerType())
    users = users.withColumn('num_years_elite', udf(users.elite))
    users = users.drop('elite')

    # assemble vector + standardize
    feat_cols = [c for c in users.columns if c not in ['user_id', 'user_id_int']]
    assembler = VectorAssembler(inputCols=feat_cols,outputCol="vector")
    scaler = StandardScaler(inputCol="vector", outputCol="features_unnormed",
                            withStd=True, withMean=True)
    users = assembler.transform(users)
    users = scaler.fit(users).transform(users)
    normalizer = Normalizer(inputCol="features_unnormed", outputCol="features")
    users = normalizer.transform(users)
    users = users.select('user_id', 'user_id_int', 'average_stars', 'features')
    return users

def user_similarities_for_collab(users):
    # vectors already normalized during preprocessing
    udf = UserDefinedFunction(lambda arr: float(arr[0].dot(arr[1])), DoubleType())
    users2 = users.select(*(col(x).alias(x + '_2') for x in users.columns))
    users2.cache()
    similarities = users.join(users2, col('user_id') != col('user_id_2'))
    similarities = similarities.withColumn("similarity_unnormed", udf(array(similarities.features, similarities.features_2)))
    mean_sim, sttdev_sim = similarities.select(mean("similarity_unnormed"), stddev("similarity_unnormed")).first()
    similarities = similarities.withColumn("similarity", (col("similarity_unnormed") - mean_sim) / sttdev_sim)
    return similarities

def get_recommendations_for_user(similarities, users, reviews):
    # once we change this to take in target_user, get rid of following line:
    target_user = users.select('user_id').collect()[0]['user_id']
    
    target_avg_stars = users.where(col('user_id') == target_user).select('average_stars').collect()[0]['average_stars']
    top_users = similarities.where(col('user_id') == target_user).sort('similarity', ascending=False)
    top_users = top_users.select('user_id_2','average_stars_2','similarity').cache()
    reviews = reviews.withColumnRenamed('user_id','rev_user_id').cache()
    similar_ratings = reviews.join(top_users, 
                               reviews.rev_user_id == top_users.user_id_2) \
                      .filter(col('stars') > 3.0)
    similar_ratings = similar_ratings.withColumn('score',(col('stars')-col('average_stars_2'))*col('similarity'))
    similar_ratings = similar_ratings.select('business_id','score','user_id_2', 'similarity')
    similar_ratings.cache()
    similar_ratings = similar_ratings.withColumn('abs_similarity', _abs(col('similarity')))
    similar_ratings = similar_ratings.groupBy("business_id").agg(_sum("score").alias("sum_score"),_sum("abs_similarity").alias("sum_similarity"))
    similar_ratings = similar_ratings.withColumn("final_score", target_avg_stars + (col("sum_score")/(col("sum_similarity"))))
    recs = similar_ratings.sort('final_score', ascending=False).limit(10)
    return recs

if __name__ == '__main__':
    spark = SparkSession.builder \
    .config("spark.driver.memory","8g") \
    .appName('yelp-recs') \
    .getOrCreate()

    user = ['User Data', user_json_path]
    userDF = readfile(user)
    review = ['Review Data', review_json_path]
    reviews = readfile(review)

    print("Start preprocessing=\t{0}".format(time.ctime(time.time())))
    users = preprocess_user_data(userDF).cache().sort(col('user_id_int'), ascending=True).limit(1000)
    users.take(1)
    print("Done preprocessing=\t{0}".format(time.ctime(time.time())))

    print("Start similarities=\t{0}".format(time.ctime(time.time())))
    similarities = user_similarities_for_collab(users).cache()
    similarities.take(1)
    print("Done similarities=\t{0}".format(time.ctime(time.time())))
    # once we get rid of sort/limit on users, pass in target_user to following function
    print("Start recs=\t{0}".format(time.ctime(time.time())))
    recs = get_recommendations_for_user(similarities, users, reviews).cache()
    recs.show(10)
    print("Done recs=\t{0}".format(time.ctime(time.time())))

    spark.stop()

