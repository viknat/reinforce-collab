import pyspark as ps 
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


class GithubCollaborativeFiltering(object):

    def __init__(self, fname='sample_gh_events.csv'):
        self.sc = ps.SparkContext('local[4]')
        self.hiveContext = ps.HiveContext(self.sc)
        self.fname = fname

    def read_data(self):
        repos_csv = self.sc.textFile(self.fname)
        header_and_rows = repos_csv.map(lambda line: line.split(','))
        header = header_and_rows.first()
        self.data = header_and_rows.filter(lambda line: line != header)

        data_fields = [StructField("repo_url", StringType(), True),
                       StructField("user_name", StringType(), True)]
        self.df = self._create_data_frame(self.data, data_fields)

    def _create_data_frame(self, rdd, fields):
        """
        OUTPUT: A Spark Data Frame 
        """
  
        schema = StructType(fields)
        return self.hiveContext.createDataFrame(rdd, schema)

    def _create_unique_ids(self):
        users_rdd = self.data.groupBy(lambda row: row[1])\
                       .zipWithUniqueId()\
                       .map(lambda x: (x[0][0], x[1]))

        users_fields = [StructField("user_name", StringType(), True),
                       StructField("user_id", IntegerType(), True)]
        self.users_df = self._create_data_frame(users_rdd, users_fields)

        repo_rdd = self.data.groupBy(lambda row: row[0])\
                       .zipWithUniqueId()\
                       .map(lambda x: (x[0][0], x[1]))

        repo_fields = [StructField("repo_url", StringType(), True),
                       StructField("repo_id", IntegerType(), True)]
        self.repos_df = self._create_data_frame(repo_rdd, repo_fields)

    def join_dfs(self):
        self._create_unique_ids()
        self.df = self.df.join(self.users_df, "user_name") \
                                .join(self.repos_df, "repo_url")

    def train_als(self):
        self.ratings = self.df.select("user_id", "repo_id")\
            .map(lambda x: Rating(x[0], x[1], 1.0))
        
        rank = 10
        numIterations = 20
        model = ALS.trainImplicit(self.ratings, rank, numIterations, alpha=0.01)

        testdata = self.ratings.map(lambda p: (p[0], p[1]))
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = self.ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
        print("Mean Squared Error = " + str(MSE))

        model.save(self.sc, "ALS_model")

    def run(self):
        self.read_data()
        self.join_dfs()
        self.train_als()

if __name__ == '__main__':
    obj = GithubCollaborativeFiltering()
    obj.run()
