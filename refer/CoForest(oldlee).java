package coforest;

/**
 * Description: CoForest is a semi-supervised algorithm, which exploits the power of ensemble learning and available
 *              large amount of unlabeled data to produce hypothesis with better performance.
 *
 * Reference:   M. Li, Z.-H. Zhou. Improve computer-aided diagnosis with machine learning techniques using undiagnosed
 *              samples. IEEE Transactions on Systems, Man and Cybernetics - Part A: Systems and Humans, 2007, 37(6).
 *
 * ATTN:        This package is free for academic usage. You can run it at your own risk.
 *	     	For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
 *
 * Requirement: To use this package, the whole WEKA environment (ver 3.4) must be available.
 *	        refer: I.H. Witten and E. Frank. Data Mining: Practical Machine Learning
 *		Tools and Techniques with Java Implementations. Morgan Kaufmann,
 *		San Francisco, CA, 2000.
 *
 * Data format: Both the input and output formats are the same as those used by WEKA.
 *
 * ATTN2:       This package was developed by Mr. Ming Li (lim@lamda.nju.edu.cn). There
 *		is a ReadMe file provided for roughly explaining the codes. But for any
 *		problem concerning the code, please feel free to contact with Mr. Li.
 *
 */

//周志华  http://lamda.nju.edu.cn/code_CoForest.ashx
//半监督性学习:基于随机森林的协同训练
import java.io.*;
import java.text.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.trees.*;

public class CoForest
{
  /** Random Forest 
  分类模型数组
  m_classifiers[i]就是第i个分类器
  */
  protected Classifier[] m_classifiers = null;

  /** The number component 
  分类器个数
  */
  protected int m_numClassifiers = 10;

  /** The random seed */
  protected int m_seed = 1;

  /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) 
      随机森林每个树使用的特征个数
      如果设置为0 则=int(logM+1)
      M是数据集的所有特征个数
      */
  protected int m_numFeatures = 0;

  /** Final number of features that were considered in last build. */
  protected int m_KValue = 0;

  /** confidence threshold 
  置信阀值
  */
  protected double m_threshold = 0.75;

  //原始数据集个数
  private int m_numOriginalLabeledInsts = 0;



  /**
   * The constructor
   */
  public CoForest()
  {
  }


  /**
   * Set the seed for initiating the random object used inside this class
   *
   * @param s int -- The seed
   */
  public void setSeed(int s)
  {
    m_seed = s;
  }

  /**
   * Set the number of trees used in Random Forest.
   *
   * @param s int -- Value to assign to numTrees.
   */
  public void setNumClassifiers(int n)
  {
    m_numClassifiers = n;
  }

  /**
   * Get the number of trees used in Random Forest
   *
   * @return int -- The number of trees.
   */
  public int getNumClassifiers()
  {
    return m_numClassifiers;
  }

  /**
   * Set the number of features to use in random selection.
   *
   * @param n int -- Value to assign to m_numFeatures.
   */
  public void setNumFeatures(int n)
  {
    m_numFeatures = n;
  }

  /**
   * Get the number of featrues to use in random selection.
   *
   * @return int -- The number of features
   */
  public int getNumFeatures()
  {
    return m_numFeatures;
  }

  /**
   * Resample instances w.r.t the weight
   *
   * @param data Instances -- the original data set
   * @param random Random -- the random object
   * @param sampled boolean[] -- the output parameter, indicating whether the instance is sampled
   * @return Instances
   对数据集data做有放回的随机抽样采样,sampled[]表示每个样本是否被采样(不论抽样几次,主要用于找出outofbag袋外样本)
   最后返回一个采样后的新Instances(weight=1)
   采样的个数等于data数据集的个数
   */
  public final Instances resampleWithWeights(Instances data,
                                             Random random,
                                             boolean[] sampled)
  {

    //在数据集data中读取每一个样本的weight,组成weights数组
    double[] weights = new double[data.numInstances()];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = data.instance(i).weight();//由于data=labeled,所以其样本的weight均等于1
    }
    Instances newData = new Instances(data, data.numInstances());
    if (data.numInstances() == 0) {
      return newData;
    }
    double[] probabilities = new double[data.numInstances()];
    double sumProbs = 0, sumOfWeights = Utils.sum(weights);
    for (int i = 0; i < data.numInstances(); i++) {
      sumProbs += random.nextDouble();//random.nextDouble()是使用一个随机种子,随机产生一个均匀分布的(0.0,1.0)之间的数据
      probabilities[i] = sumProbs;
    }
    Utils.normalize(probabilities, sumProbs / sumOfWeights);

    // Make sure that rounding errors don't mess things up
    probabilities[data.numInstances() - 1] = sumOfWeights;
    int k = 0; int l = 0;
    sumProbs = 0;
    while ((k < data.numInstances() && (l < data.numInstances()))) {
      if (weights[l] < 0) {
        throw new IllegalArgumentException("Weights have to be positive.");
      }
      sumProbs += weights[l];
      while ((k < data.numInstances()) &&
             (probabilities[k] <= sumProbs)) {
        newData.add(data.instance(l));
        sampled[l] = true;//如果被抽样了,则设置为true
        newData.instance(k).setWeight(1);//抽取后,样本的weight=1
        k++;
      }
      l++;
    }
    return newData;
  }

  /**
   * Returns the probability label of a given instance
   *
   * @param inst Instance -- The instance
   * @return double[] -- The probability label
   * @throws Exception -- Some exception
   分类器组m_classifiers预测一个样本inst,给出该样本属于每个分类的概率
   具体
   对于一个样本(不是数据集),使用m_classifiers中的每个分类器进行预测,给出该样本属于每个分类的概率
   并用所有分类器给出的概率和再归一化(0,1)得到最后的概率
   */
  public double[] distributionForInstance(Instance inst) throws Exception
  {
    double[] res = new double[inst.numClasses()];
    for(int i = 0; i < m_classifiers.length; i++)//遍历每一个分类器
    { 
      double[] distr = m_classifiers[i].distributionForInstance(inst);//distr是该样本属于每个分类的概率
      for(int j = 0; j < res.length; j++)//累加所有分类器的概率
        res[j] += distr[j];
    }
    Utils.normalize(res);//归一化
    return res;
  }

  /**
   * Classifies a given instance
   *
   * @param inst Instance -- The instance
   * @return double -- The class value
   * @throws Exception -- Some Exception
   把概率转化为预测值,取max的index
   */
  public double classifyInstance(Instance inst) throws Exception
  {
    double[] distr = distributionForInstance(inst);
    return Utils.maxIndex(distr);
  }

  /**
   * Build the classifiers using Co-Forest algorithm
   *
   * @param labeled Instances -- Labeled training set
   * @param unlabeled Instances -- unlabeled training set
   * @throws Exception -- certain exception
   */
  public void buildClassifier(Instances labeled, Instances unlabeled) throws Exception
  {
    double[] err = new double[m_numClassifiers];//本轮迭代中每个分类器的err（论文中的eit）
    double[] err_prime = new double[m_numClassifiers];//前一次迭代的err,默认为0.5（论文中的eit-1）
    double[] s_prime = new double[m_numClassifiers];//（论文中的Wit-1）即上一次迭代中新增加的已标记数据集Li的weight之和（=Li[i].sumOfWeights()）

    boolean[][] inbags = new boolean[m_numClassifiers][];//本轮迭代中,每个分类器中,每个样本是否被抽样了

    Random rand = new Random(m_seed);
    m_numOriginalLabeledInsts = labeled.numInstances();

    RandomTree rTree = new RandomTree();//新建一个随机森林模型

    // set up the random tree options
    m_KValue = m_numFeatures;
    if (m_KValue < 1) m_KValue = (int) Utils.log2(labeled.numAttributes())+1;
    rTree.setKValue(m_KValue);//设置随机森林内每个树使用的特征个数

    m_classifiers = Classifier.makeCopies(rTree, m_numClassifiers);//复制m_numClassifiers个随机森林
    Instances[] labeleds = new Instances[m_numClassifiers];//每个分类器,使用的已标记数据集labeled中抽样出来的数据集,weight均=1,且不会改变
    int[] randSeeds = new int[m_numClassifiers];//每个分类器使用的随机种子

    for(int i = 0; i < m_numClassifiers; i++)//遍历每一个分类器i
    {
      randSeeds[i] = rand.nextInt();
      ((RandomTree)m_classifiers[i]).setSeed(randSeeds[i]);
      inbags[i] = new boolean[labeled.numInstances()];//默认为false
      labeleds[i] = resampleWithWeights(labeled, rand, inbags[i]);//在原始的已标记数据集labeled,做有放回的随机抽样放于labeleds[i]中,如果抽取了一个样本,则对应的inbags[i]中的标识为true，每个样本的weight均=1
      m_classifiers[i].buildClassifier(labeleds[i]);//使用已标记数据集的抽样数据labeleds[i],训练第i个分类器
      err_prime[i] = 0.5;
      s_prime[i] = 0;
    }

    boolean bChanged = true;//是否变化
    while(bChanged)//不断迭代,直到不再变化
    { 
      bChanged = false;
      boolean[] bUpdate = new boolean[m_classifiers.length];//每个分类器i是否要再次更新(重新训练模型)
      Instances[] Li = new Instances[m_numClassifiers];//用于更新第i个分类器的新已标记样本数据集Li[i],且weight不再等于1,而是等于样本可以成为已标记样本的置信度//注意每次迭代会清空,而err_prime[i]=本次新增已标记样本数据集Li[i]的weight之和,所以err_prime每次迭代不会累加,每次迭代要重新计算
      
      for(int i = 0; i < m_numClassifiers; i++)//遍历每个分类器
      {
        err[i] = measureError(labeled, inbags, i);//使用标记样本labeled(中的袋外样本),衡量第i个分类器的预测错误率
        Li[i] = new Instances(labeled, 0);//Li[i]继承了labeled的属性,但是初始化的容量=0,即没有数据被清空了

        /** if (e_i < e'_i) */
        if(err[i] < err_prime[i])//如果分类器i的预测效果好于了上一次的预测效果则继续执行
        {
          if(s_prime[i] == 0)//第一次迭代时候s_prime=？？？？？？？？？(最大取100？？？)
            s_prime[i] = Math.min(unlabeled.sumOfWeights() / 10, 100);

          /*
          Subsample U for each hi = U'=Li[i]
          为第i个分类器的更新,在未标记样本数据集unlabeled中抽取出一些样本
          放于Li[i]中
          */
          double weight = 0;
          unlabeled.randomize(rand);//把数据集打散,变换顺序,用于随机抽样的随机动作
          int numWeightsAfterSubsample = (int) Math.ceil(err_prime[i] * s_prime[i] / err[i] - 1);//抽样个数
          for(int k = 0; k < unlabeled.numInstances(); k++)//遍历每一个未标记样本数据集unlabeled的样本
          {
            weight += unlabeled.instance(k).weight();//unlabeled的每个样本的weight均=1
            if (weight > numWeightsAfterSubsample)
             break;//抽取大于numWeightsAfterSubsample个样本后直接退出循环,不再抽取了
            Li[i].add((Instance)unlabeled.instance(k).copy());//把抽样出来的非标记样本放于Li[i]中=U'
          }

          /** for every x in U' do 
          计算第i个分类器所使用的未标记样本数据集Li[i],是否可以作为第i个分类器的标记样本数据,如果不可以则在Li[i]中删除之
          最后Li[i]的意义就是在unlabeled中抽取的未标记数据中,找出的可以作为标记数据的样本.并添加标记分类,同时修改weight=置信度
          */
          for(int j = Li[i].numInstances() - 1; j > 0; j--)
          {
            Instance curInst = Li[i].instance(j);
            if(!isHighConfidence(curInst, i))       //in which the label is assigned
              Li[i].delete(j);
          }//end of j

          //判断第i个分类器是否需要更新(重新训练)
          if(s_prime[i] < Li[i].numInstances())
          {
            if(err[i] * Li[i].sumOfWeights() < err_prime[i] * s_prime[i])
              bUpdate[i] = true;
          }
        }
      }//end of for i

      //update
      Classifier[] newClassifier = Classifier.makeCopies(rTree, m_numClassifiers);
      for(int i = 0; i < m_numClassifiers; i++)
      {
        if(bUpdate[i])
        {
          double size = Li[i].sumOfWeights();

          bChanged = true;
          m_classifiers[i] = newClassifier[i];
          ((RandomTree)m_classifiers[i]).setSeed(randSeeds[i]);
          //合并新增加的标记样本数据集Li[i]和原始的标记样本数据集labeled合并后对第i个分类器做训练
          m_classifiers[i].buildClassifier(combine(labeled, Li[i]));
          err_prime[i] = err[i];
          s_prime[i] = size;
        }
      }
    }//end of while
  }


  /**
   * To judege whether the confidence for a given instance of H* is high enough,
   * which is affected by the onfidence threshold. Meanwhile, if the example is
   * the confident one, assign label to it and weigh the example with the confidence
   *
   * @param inst Instance -- The instance
   * @param idExcluded int -- the index of the individual should be excluded from H*
   * @return boolean -- true for high
   * @throws Exception - some exception
   未标记的样本inst对于第idExcluded个分类器来说,是否可以被当做已标记样本,如果可以被当做标记样本(大于阀值)则计算出标记结果放于样本的y值内,同时更新置信度=weight
   */
  protected boolean isHighConfidence(Instance inst, int idExcluded) throws Exception
  {
    double[] distr = distributionForInstanceExcluded(inst, idExcluded);//使用除第idExcluded以外的所有分类器对未标记样本inst综合预测出该样本属于每个分类的概率
    double confidence = getConfidence(distr);//找出每个分类概率中最max的概率作为置信度
    if(confidence > m_threshold)//如果置信度大于阀值
    {
      double classval = Utils.maxIndex(distr);//找出每个分类概率中第几个分类是最max的,作为分类结果
      inst.setClassValue(classval);    //assign label,把分类结果放于样本的y值内
      inst.setWeight(confidence);      //set instance weight,把置信度放于样本的weight内
      return true;
    }
    else return false;
  }


  //按照行把L数据集和Li数据集合并为Li,并最后返回Li
  //注意Li被增加了样本数
  private Instances combine(Instances L, Instances Li)
  {
    for(int i = 0; i < L.numInstances(); i++)
      Li.add(L.instance(i));

    return Li;
  }


  //计算第id个分类器,对已标记数据集data的误差(只用袋外样本的预测正确度来计算)
  private double measureError(Instances data, boolean[][] inbags, int id) throws Exception
   {
     double err = 0;
     double count = 0;
     for(int i = 0; i < data.numInstances() && i < m_numOriginalLabeledInsts; i++)
     {
       Instance inst = data.instance(i);
       double[] distr = outOfBagDistributionForInstanceExcluded(inst, i, inbags, id);

       if(getConfidence(distr) > m_threshold)
       {
         count += inst.weight();//由于data是labeled抽样的,其weight均为1
         if(Utils.maxIndex(distr) != inst.classValue())
           err += inst.weight();
       }
     }
     err /= count;
     return err;
  }

  //计算一个样本的置信度=预测每个分类的概率中的最大的概率值
  private double getConfidence(double[] p)
  {
    int maxIndex = Utils.maxIndex(p);
    return p[maxIndex];
  }

  //和distributionForInstance很像,只是：
  //使用除了第idExcluded个以外的所有分类器,来预测一个样本inst,给出该样本属于每个分类的概率
  //
  //目的
  //为了在未标记的样本数据集中为第idExcluded个分类器生成新的标记样本数据。
  //inst是未标记的样本数据集中的一个样本
  //idExcluded是第几个分类器
  //
  //具体计算:
  //对于样本inst,依次计算该样本经过除了idExcluded以外的所有分类器的每个分类的概率
  //最后对所有分类器的概率求和,并最后归一化
  private double[] distributionForInstanceExcluded(Instance inst, int idExcluded) throws Exception
  {
    double[] distr = new double[inst.numClasses()];
    for(int i = 0; i < m_numClassifiers; i++)
    {
      if(i == idExcluded)
        continue;

      double[] d = m_classifiers[i].distributionForInstance(inst);
      for(int iClass = 0; iClass < inst.numClasses(); iClass++)
        distr[iClass] += d[iClass];
    }
    Utils.normalize(distr);
    return distr;
  }

  //在distributionForInstanceExcluded的基础上
  //添加一个限制条件,即如果inst必须要是outofbag袋外数据才可以
  private double[] outOfBagDistributionForInstanceExcluded(Instance inst, int idxInst, boolean[][] inbags, int idExcluded) throws Exception
  {
    double[] distr = new double[inst.numClasses()];
    for(int i = 0; i < m_numClassifiers; i++)
    {
      if(inbags[i][idxInst] == true || i == idExcluded)
        continue;

      double[] d = m_classifiers[i].distributionForInstance(inst);
      for(int iClass = 0; iClass < inst.numClasses(); iClass++)
        distr[iClass] += d[iClass];
    }
    if(Utils.sum(distr) != 0)
      Utils.normalize(distr);
    return distr;
  }




  /**
   * The main method only for demonstrating the simple use of this package
   *
   * @param args String[]
   */
  public static void main(String[] args)
  {
    try
    {
     int seed = 0;
     int numFeatures = 0;
     Random rand = new Random(seed);
     final int NUM_CLASSIFIERS = 6;

     BufferedReader r = new BufferedReader(new FileReader("labeled.arff"));
     Instances labeled = new Instances(r);//每个样本的weight默认设置为1
     labeled.setClassIndex(labeled.numAttributes()-1);//最后一列是label列
     r.close();

     r = new BufferedReader(new FileReader("unlabeled.arff"));
     Instances unlabeled = new Instances(r);//每个样本的weight默认设置为1
     unlabeled.setClassIndex(labeled.numAttributes()-1);//最后一列是label列
     r.close();

     r = new BufferedReader(new FileReader("test.arff"));
     Instances test = new Instances(r);//每个样本的weight默认设置为1
     test.setClassIndex(labeled.numAttributes()-1);//最后一列是label列
     r.close();

     CoForest forest = new CoForest();
     forest.setNumClassifiers(NUM_CLASSIFIERS);
     forest.setNumFeatures(numFeatures);
     forest.setSeed(rand.nextInt());
     forest.buildClassifier(labeled, unlabeled);

     double err = 0;
     for(int i = 0; i < test.numInstances(); i++)
     {
       if(forest.classifyInstance(test.instance(i)) != test.instance(i).classValue())
         err++;
     }

     System.out.println("Error Rate = " + (err/test.numInstances()));

   }
   catch(Exception e)
   {
     e.printStackTrace();
   }
 }
}
