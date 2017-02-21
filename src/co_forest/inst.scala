package co_forest

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

/**
  * Created by liming on 2017/2/15.
  * 一个样本的数据结构
  */

class InstLabeledinofbag(arg1:LabeledPoint,
                          arg2:Double) extends Serializable {
  val xandy:LabeledPoint=arg1
  val weight:Double=arg2
}

class InstLabeledoutofbag(arg1:LabeledPoint,
                           arg2:Double,
                           arg3:Array[Boolean]) extends Serializable {
  val xandy:LabeledPoint=arg1
  val weight:Double=arg2
  val inbag:Array[Boolean]=arg3    //该行样本是否在第i个分类器的抽样数据集,在袋内则=true,否则=false
}

class InstUnlabeled( arg1:Vector,
                      arg2:Int,
                      arg3:Double) extends Serializable {
  val features:Vector=arg1     //特征x变量
  val predictlabel:Int=arg2    //预测的y
  val weight:Double=arg3       //weight=预测的置信度
}
