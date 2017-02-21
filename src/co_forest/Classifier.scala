package co_forest


import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector

/**
  * Created by liming on 2017/2/12.
  * overwrite
  */
class Classifier extends Serializable {
  //训练一次模型
  def buildClassifier(trainingData:RDD[LabeledPoint]): Unit ={

  }

  //预测样本(一个样本属于每个分类的概率)
  def distributionForInstance(Instance:Vector): Array[Double] ={
    Array()
  }

}







