package co_forest

import java.util.Random

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.Map

/**
  * 在使用前要重写分类器的训练模型代码以及预测一行的代码
  * 对每个分类器的变量随机没有做该部分代码,即numFeatures_perClassifier目前没有使用
  * ？？？？？？？
  */
class Co_Forest(Classifiers:Array[Classifier],
                 numFeatures:Int,
                 numClasses:Int,
                 numFeatures_perClassifier:Int,
                 threshold:Double,
                 numMaxIteration:Int,
                 init_s_prime:Double,
                 Labeled_Dataset_arg:RDD[LabeledPoint],
                 Unlabeled_Dataset_arg:RDD[Vector],
                 IsDebug:Boolean=true) extends Serializable {


  /************************************************************************************
    * 初始化和变量
    ************************************************************************************/
  /*
  * 初始化
  * */
  val CF_cofig= new Config()//配置信息
  CF_cofig.Set_Classifiers(Classifiers) //设置分类器组
  CF_cofig.Set_numClassifiers()         //设置分类器个数
  CF_cofig.Set_numFeatures(numFeatures) //设置输入特征个数
  CF_cofig.Set_numClasses(numClasses)   //设置标记分类的个数
  CF_cofig.Set_numFeatures_perClassifier(input=numFeatures_perClassifier,method = "log2")//设置每个分类器使用的特征个数--------没有使用
  CF_cofig.Set_threshold(threshold)     //设置非标记样本成为标记样本的置信度阀值
  CF_cofig.Set_numMaxIteration(numMaxIteration) //设置最大的迭代次数
  CF_cofig.Set_init_s_prime(init_s_prime)  //设置s_prime的初始化数值,例如取100(源代码是min(100,numUnlabeledDataset/10))
  val Labeled_Dataset:RDD[LabeledPoint]=Labeled_Dataset_arg//原始的标记数据集
  //Labeled_Dataset.cache()
  val numLabeledDataset:Long=Labeled_Dataset.count() //标记数据集的样本个数
  val Unlabeled_Dataset:RDD[Vector]=Unlabeled_Dataset_arg//原始的未标记数据集
  //Unlabeled_Dataset.cache()
  val numUnlabeledDataset:Long=Unlabeled_Dataset.count() //非标记数据集的样本个数
  /*
  * 需要使用的变量
  * */
  //本轮迭代中每个分类器的err（论文中的eit）
  protected val err:Array[Double]=new Array(CF_cofig.Get_numClassifiers())
  //前一次迭代的err,默认为0.5（论文中的eit-1）
  protected val err_prime:Array[Double]=new Array(CF_cofig.Get_numClassifiers())
  //（论文中的Wit）即本次迭代中新增加的已标记数据集Li的weight之和（s(i)=Li[i].sumOfWeights()）
  protected val s:Array[Double]=new Array(CF_cofig.Get_numClassifiers())
  //（论文中的Wit-1）即上一次迭代中新增加的已标记数据集Li的weight之和（s_prime(i)=s(i)）
  protected val s_prime:Array[Double]=new Array(CF_cofig.Get_numClassifiers())



  /************************************************************************************
    * spark主逻辑
    ************************************************************************************/
  /*把原始的标记数据Labeled_Dataset
  * 进行抽样,并训练模型
  * 并标识出该数据是否被抽样过,用于袋内和袋外数据的标识
  * */
  def maindeal(): Unit ={
    if(IsDebug){
      CF_cofig.printlog()
    }
    val (labeleds_inofbag,labeleds_outofbag)=resampleWithWeightsforallclassify(Labeled_Dataset)
    for(i <-0 until CF_cofig.Get_numClassifiers()){
      CF_cofig.Get_Classifiers()(i).buildClassifier(labeleds_inofbag(i).map(x=>x.xandy))
      err_prime(i) = 0.5;
      s_prime(i) = 0;
    }
    /*
    * 迭代训练所有的分类器模型
    * */
    var bChanged:Boolean = true;//是否变化
    var Iteration:Int=0
    while(bChanged & (Iteration < CF_cofig.Get_numMaxIteration())){
      if(IsDebug){
        System.out.println(" ")
        System.out.println("Iteration ="+Iteration)
        val tmp=Labeled_Dataset.map(x=>(x.label,classifyInstance(x.features).toDouble))
        val errs=tmp.filter(x=>x._1!=x._2).count().toDouble/tmp.count().toDouble
        System.out.println("Labeled_Dataset errs="+errs)
      }
      bChanged = false;
      //每个分类器i是否要再次更新(重新训练模型)
      val bUpdate:Array[Boolean] = new Array[Boolean](CF_cofig.Get_numClassifiers());//默认每个都是false
      //用于更新第i个分类器的新已标记样本数据集Li[i],且weight不再等于1,而是等于样本可以成为已标记样本的置信度//注意每次迭代会清空,而err_prime[i]=本次新增已标记样本数据集Li[i]的weight之和,所以err_prime每次迭代不会累加,每次迭代要重新计算
      val Li:Array[RDD[InstUnlabeled]]=new Array[RDD[InstUnlabeled]](CF_cofig.Get_numClassifiers())
      for(classify_i <- 0 until CF_cofig.Get_numClassifiers()){//遍历每个分类器
        if(IsDebug){
          System.out.println("\t\tclassifier ="+classify_i)
        }
        err(classify_i) = measureError(labeleds_outofbag,classify_i);//使用标记样本labeled(中的袋外样本),衡量第i个分类器的预测错误率
        if(IsDebug){
          System.out.println("\t\t\t\tclassifier "+classify_i+" outofbag err="+err(classify_i))
        }
        Li(classify_i)=null//每次迭代清空一次
        if(err(classify_i) < err_prime(classify_i)) {
          //如果分类器i的预测效果好于了上一次的预测效果则继续执行
          if (s_prime(classify_i) == 0) {
            //第一次迭代时候s_prime=？？？？？？？？？(最大取100？？？)
            //s_prime(i) = Math.min(Unlabeled_Dataset.map(x =>1.0).reduce(_ + _) / 10, 100);
            s_prime(classify_i)=CF_cofig.Get_init_s_prime()
          }
          /*
           * Subsample U for each hi = U'=Li[i]
           * 为第i个分类器的更新,在未标记样本数据集unlabeled中抽取出一些样本
           * 放于Li[i]中
           */
          var weight:Double = 0;
          val numWeightsAfterSubsample:Long = math.min(Math.ceil(err_prime(classify_i) * s_prime(classify_i)/err(classify_i) - 1).toLong,numUnlabeledDataset);//Li(i)的抽样个数 和 非标记样本数据集个数 的最小值
          if(IsDebug){
            System.out.println("\t\t\t\tclassifier "+classify_i+" sample nums ="+numWeightsAfterSubsample)
          }
          /*
           * 首次 随机抽取numWeightsAfterSubsample个样本
           * 然后
           * 计算第i个分类器所使用的未标记样本数据集Li[i],是否可以作为第i个分类器的标记样本数据,如果不可以则在Li[i]中删除之
           * 最后Li[i]的意义就是在unlabeled中抽取的未标记数据中,找出的可以作为标记数据的样本.并添加标记分类,同时修改weight=置信度
           * 注意Li[i]是copy的数据,即新建一行InstUnlabeled
           *          其中Li(i)内如果可以作为标记数据则predictlabeled和weight会被修改
           *          另外Li(i)的样本个数是小于等于numWeightsAfterSubsample的,因为有isConfident做过滤
           */
          Li(classify_i)=sampleWithWeightsforoneclassify(Dataset = Unlabeled_Dataset,
                                                         Datasetnum = numUnlabeledDataset,
                                                         samplenum = numWeightsAfterSubsample).
            mapPartitions(iter=>{
              for{
                x<-iter
                (isConfident,predictlabeled,weight)=isHighConfidence(x,classify_i) //计算非标记样本x成为标记样本的置信度,以及预测的分类结果值
                if isConfident                                            //仅筛选出可以成为标记样本的非标记样本
              }yield (new InstUnlabeled(x,predictlabeled,weight))        //新建非标记样本InstUnlabeled,这里的predictlabeled,weight都是修改好的
            })
          /*
          * 判断第i个分类器是否需要更新(重新训练)
          * 为何是s_prime(i) < Li(i).count().toDouble呢？？？？？？
          * 即判断s_prime(i)小于Li(i)的个数
          * */
          s(classify_i)=Li(classify_i).map(x=>x.weight).reduce(_+_)
          if(IsDebug){
            System.out.println("\t\t\t\tclassifier "+classify_i+" s(i) ="+s(classify_i))
          }
          val Li_i_size=Li(classify_i).count()
          if(s_prime(classify_i) < Li_i_size.toDouble){
            if(err(classify_i) * s(classify_i) < err_prime(classify_i) * s_prime(classify_i)){
              bUpdate(classify_i) = true
              if(IsDebug) {
                System.out.println("\t\t\t\tisbUpdate=true")
              }
            }else{
              if(IsDebug){
                System.out.println("\t\t\t\tisbUpdate=false")
                System.out.println("\t\t\t\t\t\terr(classify_i)="+err(classify_i))
                System.out.println("\t\t\t\t\t\ts(classify_i)="+s(classify_i))
                System.out.println("\t\t\t\t\t\terr(classify_i)*s(classify_i)="+err(classify_i) * s(classify_i))
                System.out.println("\t\t\t\t\t\terr_prime(classify_i)="+err_prime(classify_i))
                System.out.println("\t\t\t\t\t\ts_prime(classify_i)="+s_prime(classify_i))
                System.out.println("\t\t\t\t\t\terr_prime(classify_i)*s_prime(classify_i)="+err_prime(classify_i) * s_prime(classify_i))
              }
            }
          }else{
            if(IsDebug){
              System.out.println("\t\t\t\tisbUpdate=false")
              System.out.println("\t\t\t\t\t\ts_prime(classify_i)="+s_prime(classify_i))
              System.out.println("\t\t\t\t\t\tLi_i_size="+Li_i_size)
            }
          }
        }else{
          if(IsDebug) {
            System.out.println("\t\t\t\tisbUpdate=false")
          }
        }
      }
      /*
       * 重新训练所有分类器
       */
      for(classify_i<-0 until CF_cofig.Get_numClassifiers()){
        if(bUpdate(classify_i)){
          bChanged = true;
          /*合并新增加的标记样本数据集Li[i]和原始的标记样本数据集labeled合并*/
          val NewLabeledDataset=
            Labeled_Dataset.
              union(Li(classify_i).map(x=>LabeledPoint(x.predictlabel,x.features)))
          /*对第i个分类器做训练*/
          CF_cofig.Get_Classifiers()(classify_i).buildClassifier(NewLabeledDataset);
          err_prime(classify_i) = err(classify_i);
          s_prime(classify_i) = s(classify_i);
        }
      }
      Iteration += 1
    }
  }



  /************************************************************************************
    * 涉及到rdd操作的函数
    ************************************************************************************/
  /*
  *计算第id个分类器,对已标记数据集的预测误差(注意仅仅使用袋外样本的预测正确度来计算)
  * 注意没有源代码中的 i<=m_numOriginalLabeledInsts？？？？？？？？？？
  * Instances.sample(false,numLabeledDataset)
  */
  def measureError(Instances:RDD[InstLabeledoutofbag],id:Int):Double= {
    var (err,count)=Instances.
      mapPartitions(iter=>{
        var err:Double = 0;
        var count:Double = 0;
        iter.foreach(x=>{
          val distr=outOfBagDistributionForInstanceExcluded((x.xandy.features,x.inbag),id)
          if(getConfidence(distr)>CF_cofig.Get_threshold()){
            count += x.weight
            if(maxIndex(distr)!=x.xandy.label.toInt){
              err += x.weight
            }
          }
        })
        Iterator.single((err,count))
      }).reduce((x,y)=>(x._1+y._1,x._2+y._2))
    err/count
  }
  /*
  目的：
    为每个分类器,从原始标记数据集Dataset中,有放回的随机抽取出训练样本:分为袋内和袋外数据集
    袋内数据:Array[RDD[InstLabeledinofbag]] ,Array内的每个元素是第i个分类器的训练样本,
             注意每个训练集RDD[InstLabeledinofbag]的长度=原始标记数据集Dataset的长度
    袋外数据:RDD[InstLabeledinofbag]中的LabeledPoint部分完全等于原始标记数据集Dataset中的LabeledPoint
             只是在后面添加一个inbag,其中的第i个元素,如果=true则表示该行的样本属于第i个分类器的袋内数据,否则表示袋外数据
    注意 袋内外数据的weight均要等于1.0
  思路：
    首先建立labeled_tmp=RDD[(LabeledPoint,Array[Int])]
      LabeledPoint是Dataset的LabeledPoint,Array[Int]中的第i个Int是
      样本LabeledPoint做为第i个分类器的抽样数据,被抽取的个数,如果没有被抽取则=0
      LabeledPoint的行数完全=Dataset
    然后使用labeled_tmp 建立袋外数据labeleds_outofbag=RDD[InstLabeledoutofbag]
      其中InstLabeledoutofbag的xandy=labeled_tmp中的LabeledPoint,
      weight=1.0
      inbag=labeled_tmp中的Array[Int] if元素大于则表示袋内,则=true 否则(=0)=false
    最后使用labeled_tmp 建立袋内数据labeleds_inofbag=Array[RDD[InstLabeledinofbag]].对于第i个分类器而然,
      labeleds_inofbag(i)中的xandy=labeled_tmp中的LabeledPoint,并且根据根据labeled_tmp中的Array[Int]的第i个元素大小,来创建多个
      weight=1.0
  补充：抽样说明
    为了缓解内存,在每个spark的partition数据块中,把数据按照numblock再次分块
    并且每次抽取时以numblock个数据为单元做抽取
  */
  def resampleWithWeightsforallclassify(Dataset:RDD[LabeledPoint]):(Array[RDD[InstLabeledinofbag]],
                                                                     RDD[InstLabeledoutofbag])={
    /*
    * 生成labeled_tmp
    * */
    var labeled_tmp:RDD[(LabeledPoint,Array[Int])]=null         //每次对分类器遍历时的中间数据(左侧存储原始样本数据,右侧存储该样本在第i个抽样的出现次数,如果未被抽样则=0)
    var index:Int=0//样本序号i在随机抽样数据块中的相对位置
    labeled_tmp=Dataset.map(x=>(x,Array[Int]()))
    //遍历分类器
    for(classify_i <-0 until CF_cofig.Get_numClassifiers()){
      labeled_tmp=labeled_tmp.mapPartitions(iter=>{
        val numblock:Int=10000;                     //块的大小
        var randomindexblock:Map[Int,Int]=Map()     //块内每行的随机抽样的行
        var i:Int= -1;
        for((x,y)<-iter) yield {
          i+=1
          index=i%numblock
          if(index==0)
            //每numblock个样本,就重新生成numblock个有放回的随机采样数据
            randomindexblock = gen_randomindexblock(numblock)
          if(randomindexblock.contains(index))
            (x,y :+ randomindexblock(index)) //样本x被随机选中了,并且选中了randomindexblock(index)次,插在后面
          else
            (x,y :+ 0)                       //样本x未被随机选中
        }
      })
    }
    //labeled_tmp.cache()
    /*
    * 生成labeleds_inofbag和labeleds_outofbag
    * */
    //每个分类器的抽样的袋外数据:把样本抽样次数转化为是否被抽样,y(次数)>0则为true 否则为false
    val labeleds_outofbag:RDD[InstLabeledoutofbag]=labeled_tmp.map(x=> new InstLabeledoutofbag(x._1,1.0,x._2.map(y=> y>0 )))
    //每个分类器的抽样的袋内数据:根据抽样次数,某些样本会出现多次,有些样本会不出现
    val labeleds_inofbag:Array[RDD[InstLabeledinofbag]]=new Array(CF_cofig.Get_numClassifiers())
    //遍历分类器
    for(classify_i <-0 until CF_cofig.Get_numClassifiers()){
      labeleds_inofbag(classify_i)=labeled_tmp.mapPartitions(iter=>{
        for{
          x<-iter
          samplenum=x._2(classify_i)//抽样次数
          if samplenum>0 //如果抽样次数大于0
          y<-(0 until samplenum)//遍历抽样的次数,每次重复输出一次
        }yield new InstLabeledinofbag(x._1,1.0)//weight=1.0
      })
    }
    (labeleds_inofbag,labeleds_outofbag)
  }
  /*
  在[0,numblock)中有放回的随机产生numblock个数字,作为抽取样本的位置
  */
  def gen_randomindexblock(numblock:Int):Map[Int,Int]={
    val rand=new Random()      //使用当天的服务器时间作为随机化种子,重置一个随机生成器
    val result:Map[Int,Int]=Map()
    var tmp:Int= -1
    //建立随机index
    for(j<-0 until numblock){
      tmp=rand.nextInt(numblock)   //随机生成一个数据,属于[0,numblock)
      if(!result.contains(tmp)){  //如果不存在则新建
        result += (tmp -> 0)
      }
      result(tmp)+=1
    }
    result
  }
  /*
   * 对未标记样本数据集随机不放回的采样出samplenum个样本,
   * 作为后续可转化为标记样本的备用集合
   * */
  def sampleWithWeightsforoneclassify(Dataset:RDD[Vector],Datasetnum:Long,samplenum:Long):RDD[Vector]={
    Dataset.sample(withReplacement = false,
                   fraction=math.min(1.0,samplenum.toDouble/Datasetnum.toDouble),
                   seed=System.currentTimeMillis())
  }




  /************************************************************************************
    * 针对某一行的样本数据做处理
  ************************************************************************************/
  /**
    * Returns the probability label of a given instance
    * 分类器组m_classifiers预测一个样本inst,给出该样本属于每个分类的概率
    * 具体
    * 对于一个样本(不是数据集),使用m_classifiers中的每个分类器进行预测,给出该样本属于每个分类的概率
    * 并用所有分类器给出的概率和再归一化(0,1)得到最后的概率
    */
  def distributionForInstance(Instance:Vector):Array[Double]={
    var res:Array[Double] = new Array[Double](CF_cofig.Get_numClasses());
    for(classify_i <- CF_cofig.Get_Classifiers()) {//遍历每个分类器
      val distr:Array[Double] = classify_i.distributionForInstance(Instance);//distr是该样本属于每个分类的概率
      (0 until CF_cofig.Get_numClasses()).foreach(index=>res(index)+=distr(index)) //累加所有分类器的概率
    }
    normalize(res);
    res
  }
  /*
  * 把概率转化为预测值,取max的index
  */
  def classifyInstance(Instance:Vector):Int= {
    val distr:Array[Double] = distributionForInstance(Instance);
    maxIndex(distr);
  }
  /*
  * 和distributionForInstance很像,只是：
  * 使用除了第idExcluded个以外的所有分类器,来预测一个样本inst,给出该样本属于每个分类的概率
  * 目的
  * 为了在未标记的样本数据集中为第idExcluded个分类器生成新的标记样本数据。
  * inst是未标记的样本数据集中的一个样本
  * idExcluded是第几个分类器
  * 具体计算:
  * 对于样本inst,依次计算该样本经过除了idExcluded以外的所有分类器的每个分类的概率
  * 最后对所有分类器的概率求和,并最后归一化
  */
  def distributionForInstanceExcluded(Instance:Vector,idExcluded:Int):Array[Double]= {
    var res:Array[Double] = new Array[Double](CF_cofig.Get_numClasses());
    for(i <- 0 until CF_cofig.Get_numClassifiers()) {//遍历每个分类器
      if(i!=idExcluded){
        val distr:Array[Double] = CF_cofig.Get_Classifiers()(i).distributionForInstance(Instance);//distr是该样本属于每个分类的概率
        (0 until CF_cofig.Get_numClasses()).foreach(index=>res(index)+=distr(index)) //累加所有分类器的概率
      }
    }
    normalize(res);
    res
  }
  /*
  * 计算一个样本的置信度=预测每个分类的概率中的最大的概率值
  */
  def getConfidence(p:Array[Double]):Double={
    p(maxIndex(p))
  }
  /*
  * 未标记的样本inst对于第idExcluded个分类器来说,是否可以被当做已标记样本,
  * 如果可以被当做标记样本(大于阀值)则
  *   输出(true,出标记结果,置信度=weight)
  * 否则
  *   输出(false,-1,0.0)
  */
  def isHighConfidence(Instance:Vector,idExcluded:Int):(Boolean,Int,Double)= {
    val distr:Array[Double] = distributionForInstanceExcluded(Instance,idExcluded);//使用除第idExcluded以外的所有分类器对未标记样本inst综合预测出该样本属于每个分类的概率
    val confidence:Double = getConfidence(distr);//找出每个分类概率中最max的概率作为置信度
    if(confidence > CF_cofig.Get_threshold()){//如果置信度大于阀值
      val predictlabel:Int = maxIndex(distr);//找出每个分类概率中第几个分类是最max的,作为分类结果
      (true,predictlabel,confidence)//能成为标记数据集
    }else{
      (false,-1,0.0)//不能成为标记数据集
    }
  }
  /*
  * 在distributionForInstanceExcluded的基础上
  * 添加一个限制条件,即如果inst必须要是outofbag袋外数据才可以
  * 其中Instance的Vector是样本x
  *     而Array[Boolean]是表示该样本是否是第i个分类器的袋内数据
  * idExcluded是第几个分类器
  * */
  def outOfBagDistributionForInstanceExcluded(Instance:(Vector,Array[Boolean]),
                                               idExcluded:Int):Array[Double]={
    val distr:Array[Double] = new Array[Double](CF_cofig.Get_numClasses());
    //遍历每个分类器
    for(i<-0 until CF_cofig.Get_numClassifiers()){
      //样本Instance是第i个分类器的袋外数据,并且i不等于idExcluded
      if(Instance._2(i)==false & i!=idExcluded){
        val d:Array[Double]=CF_cofig.Get_Classifiers()(i).distributionForInstance(Instance._1)
        for(iClass<- 0 until CF_cofig.Get_numClasses()){
          distr(iClass)+=d(iClass)
        }
      }
    }
    if(distr.sum!= 0)
      normalize(distr);
    distr;
  }
  //归一化
  def normalize(inputs:Array[Double]):Unit={
    val max_val:Double=inputs.max
    val min_val:Double=inputs.min
    (0 until inputs.length).foreach(i=> inputs(i)=(inputs(i)-min_val)/(max_val-min_val))
  }
  //查找那个最大的那个元素
  def maxIndex(inputs:Array[Double]):Int={
    inputs.zipWithIndex.reduce((x,y)=>if(x._1>=y._1) x else y)._2
  }
}
