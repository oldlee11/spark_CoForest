package test_mnist

import com.aliyun.odps.{Odps, TableSchema}
import com.aliyun.odps.data.Record
import org.apache.spark.SparkContext
import org.apache.spark.odps.OdpsOps
import org.apache.spark.rdd.RDD

/**
  * Created by liming on 2017/2/17.
  */
object mnist_data {
  def read_mnist(sc:SparkContext,
                  odpsOps:OdpsOps,
                  odps:Odps,
                  projectName:String,
                  input_odps_db:String,
                  numPartition:Int=0): RDD[(Array[Double],Double)] ={
    if(!odps.tables().exists(input_odps_db)){
      System.err.println(s"odps table $projectName.$input_odps_db is not build")
      sc.stop()
    }else {
      System.out.println(s"success find table $projectName.$input_odps_db")
    }
    val pathflow_hdfs=
      odpsOps.readTable(project=projectName,
                        table=input_odps_db,
                        transfer = (r: Record, schema: TableSchema) => Array(r.getString(0),r.getString(1)),
                        numPartition=numPartition).map(x=>(x(0).split(",").map(y=>y.toDouble),x(1).toDouble))
    System.out.println(s"success read table $projectName.$input_odps_db")
    pathflow_hdfs
  }

}
