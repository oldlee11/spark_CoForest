package test_mnist

import co_forest.{Classifier, Co_Forest}
import com.aliyun.odps.cupid.CupidSession
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.odps.OdpsOps
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by liming on 2017/2/20.
  */
object Co_Forest_LR {
  def main(args: Array[String]):Unit={
    val numPartition=args(0).toInt
    val numClassifiers=args(1).toInt
    val init_s_prime=args(2).toDouble
    val conf = new SparkConf().setAppName("Co_Forest_LR_for_mnist")
    val sc = new SparkContext(conf)
    val odps = CupidSession.get.odps
    val odpsOps = OdpsOps(sc)
    val projectName = conf.get("project_name")

    //读取数据集(不用标准化预处理)
    val DataSetTraining=
      mnist_data.read_mnist(sc,odpsOps,odps,projectName,"lm_mnist_dataset_training",numPartition).
        map(x=>LabeledPoint(x._2,Vectors.dense(values = x._1)));
    val splits=DataSetTraining.randomSplit(Array(0.1, 0.9))
    val DataSetLabel=splits(0)
    val DataSetUnlabel=splits(1).map(x=>x.features)
    val DataSetLabelLen=DataSetLabel.count()
    val DataSetUnlabelLen=DataSetUnlabel.count()
    System.out.println("DataSetLabel set len="+DataSetLabelLen.toString())
    System.out.println("DataSetUnlabel set len="+DataSetUnlabelLen.toString())
    DataSetLabel.cache()
    DataSetUnlabel.cache()
    val DataSetTest=
      mnist_data.read_mnist(sc,odpsOps,odps,projectName,"lm_mnist_dataset_test",numPartition).
        map(x=>LabeledPoint(x._2,Vectors.dense(values = x._1)));


    //建立co_forest
    val classifiers_imp:Array[Classifier]=new Array(numClassifiers)
    for(i<-0 until numClassifiers){
      classifiers_imp(i)=new Classifier_imp()
    }
    val CoForestmod=new Co_Forest(Classifiers = classifiers_imp ,
                   numFeatures=784,
                   numClasses=10,
                   numFeatures_perClassifier=0,
                   threshold=0.75,
                   numMaxIteration=100,
                   init_s_prime=init_s_prime,
                   Labeled_Dataset_arg=DataSetLabel,
                   Unlabeled_Dataset_arg=DataSetUnlabel)
    CoForestmod.maindeal()

  }
}

/*****************************************
success find table ytanalyst_dev.lm_mnist_dataset_training
success read table ytanalyst_dev.lm_mnist_dataset_training
partitionSizeInstance id = 20170221050223359g2mqafpe
DataSetLabel set len=5066
DataSetUnlabel set len=44934
success find table ytanalyst_dev.lm_mnist_dataset_test
success read table ytanalyst_dev.lm_mnist_dataset_test

numClassifiers=5
numFeatures=784
numFeatures_perClassifier=11
threshold=0.75
numMaxIteration=100

Iteration =0
Labeled_Dataset errs=0.021713383339913146
		classifier =0
				classifier 0 outofbag err=0.044566253574833174
				classifier 0 sample nums =1121
				classifier 0 s(i) =1096.0
				isbUpdate=true
		classifier =1
				classifier 1 outofbag err=0.05050984111927911
				classifier 1 sample nums =989
				classifier 1 s(i) =996.0
				isbUpdate=false
						err(classify_i)=0.05050984111927911
						s(classify_i)=996.0
						err(classify_i)*s(classify_i)=50.30780175480199
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =2
				classifier 2 outofbag err=0.051585423568386184
				classifier 2 sample nums =969
				classifier 2 s(i) =987.0
				isbUpdate=false
						err(classify_i)=0.051585423568386184
						s(classify_i)=987.0
						err(classify_i)*s(classify_i)=50.914813061997165
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =3
				classifier 3 outofbag err=0.052029864675688285
				classifier 3 sample nums =960
				classifier 3 s(i) =987.0
				isbUpdate=false
						err(classify_i)=0.052029864675688285
						s(classify_i)=987.0
						err(classify_i)*s(classify_i)=51.35347643490434
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =4
				classifier 4 outofbag err=0.04728132387706856
				classifier 4 sample nums =1057
				classifier 4 s(i) =999.0
				isbUpdate=true

Iteration =1
Labeled_Dataset errs=0.0025661271219897357
		classifier =0
				classifier 0 outofbag err=0.03163361661945231
				classifier 0 sample nums =1544
				classifier 0 s(i) =1577.0
				isbUpdate=false
						err(classify_i)=0.03163361661945231
						s(classify_i)=1577.0
						err(classify_i)*s(classify_i)=49.886213408876294
						err_prime(classify_i)=0.044566253574833174
						s_prime(classify_i)=1096.0
						err_prime(classify_i)*s_prime(classify_i)=48.84461391801716
		classifier =1
				classifier 1 outofbag err=0.02353494939985879
				classifier 1 sample nums =2124
				classifier 1 s(i) =2172.0
				isbUpdate=false
						err(classify_i)=0.02353494939985879
						s(classify_i)=2172.0
						err(classify_i)*s(classify_i)=51.1179100964933
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =2
				classifier 2 outofbag err=0.022700401986285174
				classifier 2 sample nums =2202
				classifier 2 s(i) =2185.0
				isbUpdate=true
		classifier =3
				classifier 3 outofbag err=0.020455602045560205
				classifier 3 sample nums =2444
				classifier 3 s(i) =2454.0
				isbUpdate=false
						err(classify_i)=0.020455602045560205
						s(classify_i)=2454.0
						err(classify_i)*s(classify_i)=50.198047419804745
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =4
				classifier 4 outofbag err=0.03583138173302108
				classifier 4 sample nums =1318
				classifier 4 s(i) =1261.0
				isbUpdate=true

Iteration =2
Labeled_Dataset errs=0.0
		classifier =0
				classifier 0 outofbag err=0.01941974730931212
				classifier 0 sample nums =2515
				classifier 0 s(i) =2525.0
				isbUpdate=false
						err(classify_i)=0.01941974730931212
						s(classify_i)=2525.0
						err(classify_i)*s(classify_i)=49.0348619560131
						err_prime(classify_i)=0.044566253574833174
						s_prime(classify_i)=1096.0
						err_prime(classify_i)*s_prime(classify_i)=48.84461391801716
		classifier =1
				classifier 1 outofbag err=0.008557166627050155
				classifier 1 sample nums =5843
				classifier 1 s(i) =5913.0
				isbUpdate=false
						err(classify_i)=0.008557166627050155
						s(classify_i)=5913.0
						err(classify_i)*s(classify_i)=50.598526265747566
						err_prime(classify_i)=0.5
						s_prime(classify_i)=100.0
						err_prime(classify_i)*s_prime(classify_i)=50.0
		classifier =2
				classifier 2 outofbag err=0.02429245283018868
		classifier =3
				classifier 3 outofbag err=0.011108484991727724
				classifier 3 sample nums =4501
				classifier 3 s(i) =4349.0
				isbUpdate=true
		classifier =4
				classifier 4 outofbag err=0.02203469292076887
				classifier 4 sample nums =2050
				classifier 4 s(i) =1977.0
				isbUpdate=true

Iteration =3
Labeled_Dataset errs=0.0
		classifier =0
				classifier 0 outofbag err=0.00942507068803016
				classifier 0 sample nums =5182
				classifier 0 s(i) =5214.0
				isbUpdate=false
						err(classify_i)=0.00942507068803016
						s(classify_i)=5214.0
						err(classify_i)*s(classify_i)=49.14231856738925
						err_prime(classify_i)=0.044566253574833174
						s_prime(classify_i)=1096.0
						err_prime(classify_i)*s_prime(classify_i)=48.84461391801716
		classifier =1
				classifier 1 outofbag err=0.0
				classifier 1 sample nums =44934
				classifier 1 s(i) =44934.0
				isbUpdate=true
		classifier =2
				classifier 2 outofbag err=0.01118781242561295
				classifier 2 sample nums =4433
				classifier 2 s(i) =4548.0
				isbUpdate=false
						err(classify_i)=0.01118781242561295
						s(classify_i)=4548.0
						err(classify_i)*s(classify_i)=50.882170911687695
						err_prime(classify_i)=0.022700401986285174
						s_prime(classify_i)=2185.0
						err_prime(classify_i)*s_prime(classify_i)=49.6003783400331
		classifier =3
				classifier 3 outofbag err=0.01050420168067227
				classifier 3 sample nums =4599
				classifier 3 s(i) =4481.0
				isbUpdate=true
		classifier =4
				classifier 4 outofbag err=0.010397412199630314
				classifier 4 sample nums =4189
				classifier 4 s(i) =4194.0
				isbUpdate=false
						err(classify_i)=0.010397412199630314
						s(classify_i)=4194.0
						err(classify_i)*s(classify_i)=43.60674676524954
						err_prime(classify_i)=0.02203469292076887
						s_prime(classify_i)=1977.0
						err_prime(classify_i)*s_prime(classify_i)=43.56258790436005

Iteration =4
Labeled_Dataset errs=0.0
		classifier =0
				classifier 0 outofbag err=2.3430178069353328E-4
				classifier 0 sample nums =44934
				classifier 0 s(i) =44934.0
				isbUpdate=true
		classifier =1
				classifier 1 outofbag err=0.0
		classifier =2
				classifier 2 outofbag err=4.7404598246029864E-4
				classifier 2 sample nums =44934
				classifier 2 s(i) =44934.0
				isbUpdate=true
		classifier =3
				classifier 3 outofbag err=0.0
				classifier 3 sample nums =44934
				classifier 3 s(i) =44934.0
				isbUpdate=true
		classifier =4
				classifier 4 outofbag err=0.0
				classifier 4 sample nums =44934
				classifier 4 s(i) =44934.0
				isbUpdate=true

Iteration =5
Labeled_Dataset errs=0.0017765495459928938
		classifier =0
				classifier 0 outofbag err=0.0018674136321195146
		classifier =1
				classifier 1 outofbag err=0.003058823529411765
		classifier =2
				classifier 2 outofbag err=0.0027945971122496508
		classifier =3
				classifier 3 outofbag err=0.001876172607879925
		classifier =4
				classifier 4 outofbag err=0.0018885741265344666
 *****************************************/