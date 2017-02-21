package test_mnist

import org.apache.spark.rdd.RDD

/**
  * Created by liming on 2017/2/20.
  */
object predmodle {
  def labelAndPreds_sum(labelAndPreds:RDD[(Double,Double)]): Unit ={
    val lens=labelAndPreds.count().toDouble
    val errs=labelAndPreds.filter(x=>x._1!=x._2).count().toDouble
    System.out.println("err is="+ errs/lens)
    /*val errs_bykey=labelAndPreds.filter(x=>x._1!=x._2).countByKey().map(x=>Array(x._1,x._2.toDouble/lens).mkString(":")).toArray
    errs_bykey.foreach(x=>System.out.println(x))*/
  }
}
