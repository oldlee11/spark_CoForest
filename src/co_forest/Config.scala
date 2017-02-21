package co_forest

/**
  * Created by liming on 2017/2/12.
  * 半监督性学习框架
  * 基于随机森林的互协同训练
  */
class Config extends Serializable {
  /*
  * 一组分类器(分类器森林)
  */
  protected var Classifiers:Array[Classifier]=Array()
  def Set_Classifiers(input:Array[Classifier]):Unit={
    Classifiers=input
  }
  def Get_Classifiers(): Array[Classifier] ={
    Classifiers
  }
  protected var numClassifiers:Int=0
  def Set_numClassifiers(input:Int=0):Unit={
    numClassifiers=if(input==0) Classifiers.length else input //默认=m_classifiers.length
  }
  def Get_numClassifiers(): Int ={
    numClassifiers
  }

  /*
  * 标记的y值可以取的分类个数
  * */
  protected var numClasses:Int=0
  def Set_numClasses(input:Int): Unit ={
    numClasses=input
  }
  def Get_numClasses():Int={
    numClasses
  }

  /*
  * 样本的特征个数
  * */
  protected var numFeatures:Int=0
  def Set_numFeatures(input:Int)={
    numFeatures=input
  }
  def Get_numFeatures():Int={
    numFeatures
  }

  /*
  * 每个分类器使用的特征个数
  * */
  protected var numFeatures_perClassifier:Int=0
  def Set_numFeatures_perClassifier(input:Int=0,method:String="log2"): Unit ={
    if(input==0){
      if(method=="log2"){
        numFeatures_perClassifier=math.ceil(math.log(numFeatures)/math.log(2)+1).toInt;
      }
    }else{
      numFeatures_perClassifier=input
    }
  }
  def Get_numFeatures_perClassifier():Int={
    numFeatures_perClassifier
  }

  /*
  * 作为标记样本的置信度阀值
  * */
  protected var threshold:Double=0
  def Set_threshold(input:Double): Unit ={
    threshold=if(input==0) 0.75 else input //默认=0.75
  }
  def Get_threshold():Double={
    threshold
  }

  /*
  * 最大的迭代次数
  * */
  protected var numMaxIteration:Int=0
  def Set_numMaxIteration(input:Int)={
    numMaxIteration=input
  }
  def Get_numMaxIteration():Int={
    numMaxIteration
  }

  /*
  * 默认的init_s_prime
  * */
  protected var init_s_prime:Double=0
  def Set_init_s_prime(input:Double)={
    init_s_prime=input
  }
  def Get_init_s_prime():Double={
    init_s_prime
  }

  /*
  * 配置的print
  * */
  def printlog(): Unit ={
    System.out.println(" ")
    System.out.println("numClassifiers="+Get_numClassifiers())
    System.out.println("numFeatures="+Get_numFeatures())
    System.out.println("numFeatures_perClassifier="+Get_numFeatures_perClassifier())
    System.out.println("threshold="+Get_threshold())
    System.out.println("numMaxIteration="+Get_numMaxIteration())
  }

}
