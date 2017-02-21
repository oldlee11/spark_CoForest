package test_mnist

import co_forest.Classifier
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by liming on 2017/2/20.
  */
class Classifier_imp extends Classifier {
  var solModel_lr:LogisticRegressionModel=null
  //训练一次模型
  override def buildClassifier(trainingData:RDD[LabeledPoint]): Unit ={
    solModel_lr=new LogisticRegressionWithLBFGS().setNumClasses(10).run(trainingData)
  }

  //预测样本(一个样本属于每个分类的概率,这里没能给出概率,只给出了1.0和0.0)
  override def distributionForInstance(Instance:Vector): Array[Double] ={
    val index:Int=solModel_lr.predict(Instance).toInt
    val result:Array[Double]=new Array(10)
    result(index)=1.0
    result
  }
}
