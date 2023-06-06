from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark=SparkSession.builder.appName('practice').getOrCreate()
data=spark.read.option('header','true').csv('2019.csv',inferSchema=True)
feature=VectorAssembler(inputCols=['GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices','Perceptions of corruption','Generosity'],outputCol='Independent Features')
output=feature.transform(data)
finalized_output=output.select('Independent Features','Score')
train,test=finalized_output.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features',labelCol='Score')
regressor=regressor.fit(train)
pred_result=regressor.evaluate(test)
pred_result.predictions.show()
print(pred_result.meanAbsoluteError,pred_result.meanSquaredError)



