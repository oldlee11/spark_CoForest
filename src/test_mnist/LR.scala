package test_mnist


import com.aliyun.odps.cupid.CupidSession
import org.apache.spark._
//import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.odps.OdpsOps


/**
  * Created by liming on 2017/2/17.
  */
object LR {
  def main(args: Array[String]):Unit={
    val numPartition=args(0).toInt
    val conf = new SparkConf().setAppName("LR_for_mnist")
    val sc = new SparkContext(conf)
    val odps = CupidSession.get.odps
    val odpsOps = OdpsOps(sc)
    val projectName = conf.get("project_name")


    //读取数据集(不用标准化预处理)
    val DataSetTraining=
      mnist_data.read_mnist(sc,odpsOps,odps,projectName,"lm_mnist_dataset_training",numPartition).
      map(x=>LabeledPoint(x._2,Vectors.dense(values = x._1)));
    val traindatasetlen=DataSetTraining.count()
    System.out.println("train data set len="+traindatasetlen.toString())
    DataSetTraining.cache()
    val DataSetTest=
      mnist_data.read_mnist(sc,odpsOps,odps,projectName,"lm_mnist_dataset_test",numPartition).
      map(x=>LabeledPoint(x._2,Vectors.dense(values = x._1)));

    //训练模型
    val solModel_lr=new LogisticRegressionWithLBFGS().setNumClasses(10).run(DataSetTraining)
    //val solModel_lr=LogisticRegressionWithSGD.train(DataSetTraining,numIterations)
    val labelAndPreds = DataSetTest.map( x =>(x.label,solModel_lr.predict(x.features)))
    predmodle.labelAndPreds_sum(labelAndPreds)    // 7.55%

  }
}
