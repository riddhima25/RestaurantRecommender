import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StandardScaler, Normalizer
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.sql.functions import UserDefinedFunction, array, mean, stddev
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.recommendation import ALS
from  pyspark.sql.functions import abs as _abs
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

def get_recommendations(users, reviews, businesses): 
    target_user_entry = users.select('user_id', 'average_stars').collect()[0]
    target_user = target_user_entry['user_id']
    target_user_avg_stars = target_user_entry['average_stars']

    user_reviews = reviews.select('user_id','business_id','stars').where(col('user_id') == target_user)
    target_user_total_stars = user_reviews.groupBy('user_id').agg(_sum("stars").alias('total_stars')) \
                                .select('total_stars').collect()[0]['total_stars']
    other_user_reviews = reviews.select('user_id','business_id','stars').where(col('user_id') != target_user)
    other_user_reviews = other_user_reviews.select(*(col(x).alias(x + '_2') for x in other_user_reviews.columns))

    common_bus = user_reviews.join(other_user_reviews, user_reviews.business_id == other_user_reviews.business_id_2) \
                    .select('user_id', 'user_id_2', 'business_id', 'stars', 'stars_2')
    common_bus = common_bus.join(businesses, common_bus.business_id == businesses.business_id) \
                    .withColumn('total_rating', businesses.stars * businesses.review_count) \
                    .select(common_bus.user_id.alias('user_id_1'), 'user_id_2', businesses.business_id, common_bus.stars.alias('stars_1'), 'stars_2', 'total_rating')

    recopower = common_bus.withColumn('ind_power', (col('stars_1'))*(col('stars_2')) / (target_user_total_stars * (col('total_rating')))) \
                    .groupBy("user_id_2").agg(_sum("ind_power").alias("power"))
    recopower = recopower.join(users, recopower.user_id_2 == users.user_id) \
                    .select(recopower.columns + ['average_stars'])

    similar_ratings = recopower.join(reviews, recopower.user_id_2 == reviews.user_id) \
                        .select(recopower.columns + ['business_id', 'stars'])
    similar_ratings = similar_ratings.join(user_reviews, similar_ratings.business_id != user_reviews.business_id, "leftsemi")

    recs = similar_ratings.withColumn("ind_rating", target_user_avg_stars + (col('power')) * ((col('stars')) - (col('average_stars')))) \
                        .groupBy('business_id').agg(_sum("ind_rating").alias('pred_rating'))
    recs = recs.sort('pred_rating', ascending=False).cache()
    return recs


if __name__ == '__main__':
    spark = SparkSession.builder \
    .config("spark.driver.memory","8g") \
    .appName('yelp-recs') \
    .config("spark.sql.broadcastTimeout", "1200") \
    .getOrCreate()

    user = ['User Data', user_json_path]
    users = readfile(user).filter((col('review_count') >= 10) & (col('average_stars') >= 3.0)).limit(100).cache()

    review = ['Review Data', review_json_path]
    reviews = readfile(review)
    reviews = reviews.join(users, reviews.user_id == users.user_id, "leftsemi").cache()

    business = ['Business Data', business_json_path]
    businesses = readfile(business)
    businesses = businesses.join(reviews, businesses.business_id == reviews.business_id, "leftsemi").cache()

    recs = get_recommendations(users, reviews, businesses)
    recs.show(10)

    spark.stop()