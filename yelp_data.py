# Whatever you don't have, install (pip or conda)
%matplotlib inline
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import time
import geopy
from geopy.distance import geodesic
import geopandas as gpd
import shapely
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('Yelp Data').config("spark.some.config.option", "some-value").getOrCreate() # For dataframes

# Paths for reading JSON, change to whatever your root is 
business_json_path = , 'yelp_academic_dataset_business.json']  
user_json_path = 'yelp_academic_dataset_user.json'
review_json_path = 'yelp_academic_dataset_review.json'
# List for printing time to execute later
business = ['Business Data', business_json_path]
user = ['User Data', user_json_path]
review = ['Review Data', review_json_path]

# Method to read JSON into Pyspark Dataframe
def readfile(datatype):
    start = time.time()   # Read in JSON as Spark Dataframe and count the time
    print("{0} Start-time=\t{1}".format(datatype[0], time.ctime(start)))
    DF = spark.read.json(datatype[1])
    end = time.time()
    print("{0} End-time=\t{1}".format(datatype[0], time.ctime(end)))
    print("Time elapsed=\t{0} s \n".format(end-start))
    return DF
    
# Read business, user, and review JSONs
busDF = readfile(business)
userDF = readfile(user)
revDF = readfile(review)

# Print Schema of each Spark dataframe, commented out, uncomment if needed
'''
busDF.printSchema()
userDF.printSchema()
revDF.printSchema()
'''

print("Total Businesses in Yelp Dataset: {0}".format(busDF.count()))

# Select restaurants and sort by latitude & longitude in ascending order
rdf = busDF.filter(busDF['is_open']==True).filter(busDF.categories.contains('Restaurant')).orderBy(['latitude','longitude'])


print("Total Open Restaurants: {0}".format(rdf.count())) # Should be 43980 total restaurants

# Convert Spark Dataframe to Pandas, to eventually find restaurants within radius
def convert_SparkDataframe_to_PandaDataframe(sparkdf):
    start = time.time()   
    print("Start-time to convert=\t{0}".format(time.ctime(start)))
    # Convert to pandas for finding neighboring restaurants within a radius, and order by latitude/longitude
    DF = sparkdf.toPandas().sort_values(by=['latitude','longitude'])
    end = time.time()
    print("End-time to convert =\t{0}".format(time.ctime(end)))
    print("Time elapsed=\t{0} s \n".format(end-start))
    return DF
    
# Convert restaurants Spark dataframe to pandas dataframe
rpdf = convert_SparkDataframe_to_PandaDataframe(rdf)

#Method to return a list with chosen restaurant ID and a geopandas dataframe of neighbors within radius
def getNeighbors(pd, radius, restaurant):
    # Print start time
    start = time.time()   
    print("Start-time to convert=\t{0}".format(time.ctime(start)))
    # Restaurant is an index of a restaurant from 0 to pd.shape[0]-1, radius is miles
    # Convert pandas dataframe to geopandas dataframe (next line)
    # crs is coordinate reference system, EPSG:4326 is standard latitude/longitude units
    gdf = gpd.GeoDataFrame(pd, geometry=gpd.points_from_xy(pd['longitude'], pd['latitude']), crs={"init":"EPSG:4326"}) 
    gdf_proj = gdf.to_crs({"init": "EPSG:3857"})
    gdf_single = gdf[restaurant:restaurant+1] # Choose single restaurant
    print("BUSINESS ID: ", pd.iloc[restaurant].business_id)
    # EPSG:3857 converts latitude/longitude units to actual metric units
    gdf_single_proj = gdf_single.to_crs({"init": "EPSG:3857"})
    # Create distance buffer to find restaurants within radius in meters, multiply miles x meters/mile (1609.34 m/mi)
    circle = gdf_single_proj.buffer(radius*1609.34).unary_union 
    # Column of restaurants with empty coordinates if not within radius
    isNeighbor = gdf_proj['geometry'].intersection(circle) 
    # Get final Geopandas dataframe with only the neighbors within the radius
    neighbors = gdf_proj[~isNeighbor.is_empty]
    # Print end time
    end = time.time()
    print("End-time to convert =\t{0}".format(time.ctime(end)))
    print("Time elapsed=\t{0} s \n".format(end-start))
    # return neighbors geopandas dataframe
    return [pd.iloc[restaurant].business_id, neighbors]

# For example's sake, we'll choose the 2001-th restaurant and find its neighbors with a 10 mile radius
neighbors2000 = getNeighbors(rpdf, 10, 2000)
print("Restaurant {0} has {1} neighbors.".format(neighbors2000[0], neighbors2000[1].shape[0])
# Now, read in the shapefiles for US states and Canadian provinces in order to map in geopandas:
usashp = './states21basic.shp' # Add file path, and you'll need all the attached files, specifically of type .dbf, .prj, .shp, and .shx, otherwise it won't work
canshp = './Canada_AL263.shp'
USA = gpd.read_file(usashp)
CAN = gpd.read_file(canshp)

# Combine USA states and Canadian provinces into 1 Geopandas dataframe
NorAm = USA.append(CAN)

# Brief data exploration, commented out
'''
USA.head()
CAN.head()
NorAm.head()
'''
# get all states within the neighbor dataset
states_in_neighbors = neighbors2000[1].state.unique()[0]
print("States with restaurants in neighbors set: ", states_in_neighbors)

# Get only states you need to map
states_to_map = NorAm[NorAm['state_abbr']==states_in_neighbors]

# Convert neighbors' locations back to conventional latitude/longitude
neighbors2000_2 = [neighbors2000[0], neighbors2000[1].to_crs({"init": "EPSG:4326"})]

# Plot the neighbors within the state(s), which is Arizona in this case
base = states_to_map.boundary.plot(figsize=[64,32])
neighbors2000_2[1].plot(ax = base, color='red',markersize=0.1)
plt.title('Restaurants within 10 miles of Business ID: {0}'.format(neighbors2000_2[0]),fontsize=32)

# Plot all restaurants in dataset in map of North America
base2 = NorAm.boundary.plot(figsize=[64,32])
allresloc = gpd.GeoDataFrame(rpdf, geometry=gpd.points_from_xy(rpdf['longitude'], rpdf['latitude']), crs={"init":"EPSG:4326"})
allresloc.plot(ax=base2, figsize=[64,32],color='red',markersize=0.5)
plt.title('Locations of Restaurants in Yelp Dataset throughout North America: {0}'.format(neighbors2000_2[0]),fontsize=32)

# Stop spark session
spark.stop()