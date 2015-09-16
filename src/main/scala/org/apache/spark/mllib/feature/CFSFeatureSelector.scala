package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

//import java.io.File
//import breeze.linalg.{CSCMatrix, csvwrite}

object CFSFeatureSelector {
  
    def select(sc: SparkContext, data: RDD[LabeledPoint], searchMethod: String): (Double, Array[Int]) =
    {
      val dataByFeature=data.flatMap({x => x.features.toArray.view.zipWithIndex.map({case (x,i) => (i, (x,x*x,1))}) :+ (Int.MaxValue,(x.label, x.label*x.label, 1))})
      val numFeatures=data.first().features.size
//dataByFeature.foreach(println)
      val stddevs=computeStdDevs(dataByFeature)
//println("STDDEVS:")
//stddevs.foreach(println)

      val correlMatrix: Matrix = computeCorrelationMatrix(data)
//println(correlMatrix)
//csvwrite(new File("/home/eirasf/Escritorio/myCSCMatrix.txt"), correlMatrix.toBreeze, separator=' ')

      val devs=stddevs.collect()
      val bCorrelMatrix=sc.broadcast(correlMatrix)

      val attribIndices=new Array[Int](1)
      attribIndices(0)=0
      
//val merit=evalSubset(attribIndices, devs, bCorrelMatrix)
//println("Merit: "+merit)

      if (searchMethod.toLowerCase()=="bestfirst")
        return bestFirstSearch(sc, numFeatures-1, devs, bCorrelMatrix)
      else
        return greedySearch(sc, numFeatures-1, devs, bCorrelMatrix)
    }
    
    def bestFirstSearch(sc: SparkContext, maxFeature: Int, devs: Array[Double], bCorrelMatrix: Broadcast[Matrix]): (Double,Array[Int]) =
    {
      val numMaxStale=5
      var candidates=List((Array[Int](),0.0))
      var stale=0
      var bestMerit=0.0
      var worstMerit=0.0
      var bestSet=Array[Int]()
      while((candidates.length>0) && (stale<numMaxStale))
      {
        val cand=candidates.head._1
        candidates=candidates.tail
        val merits=sc.parallelize(1 to maxFeature).flatMap({
            case x if (cand contains x) => None
            case x => val newSet=cand :+ x
                      Some(newSet,evalSubset(newSet, devs, bCorrelMatrix))
                      //A cache could be used to avoid evaluating the same set more than once,
                      //but the cost of getting it to the compute nodes will probably be larger
                      //than the gain obtained
          }).filter(_._2>worstMerit)
        var added=false;
        if (merits.count()>0)
        {
          val res:Array[(Array[Int],Double)]=merits.collect()
          candidates=(candidates ++ res.toList).sortBy(_._2*(-1)).take(5)
          if ((candidates.head._2>bestMerit) ||
              ((candidates.head._2==bestMerit) && (candidates.head._1.length<bestSet.length)))
          {
            bestMerit=candidates.head._2
            bestSet=candidates.head._1
            added=true
            stale=0
          }
          worstMerit=candidates.last._2
        }
        if (!added)
          stale=stale+1
          
/*println("\n\nPASO----------------")
candidates.foreach({case (l,m) => print("L:"+l.sorted.mkString(","))
                                  println(" M:"+m)})
println("Stale: "+stale)
println("Best merit: "+bestMerit)*/
      }
      return (bestMerit, bestSet)
    }
    
    def greedySearch(sc: SparkContext, maxFeature: Int, devs: Array[Double], bCorrelMatrix: Broadcast[Matrix]): (Double,Array[Int]) =
    {
      val attribIndices=Array[Int]()
      return greedySearchStep(sc, 0.0, attribIndices, maxFeature, devs, bCorrelMatrix) 
    }
    
    def greedySearchStep(sc: SparkContext, previousMerit: Double, attribIndices: Array[Int], maxFeature: Int, devs: Array[Double], bCorrelMatrix: Broadcast[Matrix]): (Double,Array[Int]) =
    {
      val featuresToTest=0 to maxFeature
      //val maxValue=featuresToTest.flatMap({
      val maxValue=sc.parallelize(featuresToTest).flatMap({
//      val values=featuresToTest.flatMap({
                        case x if (attribIndices.contains(x)) => None
                        case x => Some((x,evalSubset(attribIndices :+ x, devs, bCorrelMatrix)))
                      }).sortBy({x=>x._2*(-1)})
                      //.head()
                      .first()
/*println("\nTesting: ")
values.foreach(println)
val maxValue=values.head
println("\t: "+maxValue._2+" by adding "+maxValue._1)*/
      
      if (maxValue._2<=previousMerit)
        return (previousMerit, attribIndices)
      if (attribIndices.length>=maxFeature-2)
        return (maxValue._2, attribIndices :+ maxValue._1)
      return greedySearchStep(sc, maxValue._2, attribIndices :+ maxValue._1, maxFeature, devs, bCorrelMatrix)
    }
  
    def evalSubset(attribIndices: Array[Int], devs: Array[Double], bCorrelMatrix: Broadcast[Matrix]) : Double =
    {
      val classIndex=bCorrelMatrix.value.numCols-1;
      val fraction=attribIndices.map({x => (devs(x) * bCorrelMatrix.value(x, classIndex),
                                            attribIndices.map({case a  if (a<x) => 2.0 * devs(x) * devs(a) * bCorrelMatrix.value(x, a)
                                                                case a => 0.0})
                                                          .reduce(_+_)
                                                          + devs(x) * devs(x))})
                   .reduce({(a,b) => (a._1+b._1, a._2+b._2)})
//println(fraction)
      if (fraction._2==0.0)
        return 0.0
      return fraction._1/math.sqrt(fraction._2)
    }
    
    def computeCorrelationMatrix(data: RDD[LabeledPoint]) : Matrix =
    {
      return Statistics.corr(data.map({ x => Vectors.dense(x.features.toArray :+ x.label) }), "pearson")//"spearman")
                                           .map(math.abs(_))
    }
    
    def computeStdDevs(dataByFeature: RDD[(Int, (Double, Double, Int))]): RDD[Double] =
    {
      return dataByFeature.reduceByKey({case ((s1,ss1,c1),(s2,ss2,c2)) => (s1+s2,ss1+ss2,c1+c2)})
                             .map({case (i,(sum, sumSquares, count)) =>
                                         val mean=sum/count
                                         //(i,mean*mean/count+sumSquares-2*mean*sum)})//Todas estas igualdades van sin Math.sqrt
                                         //(i,mean*sum+sumSquares-2*mean*sum)})
                                         //(i,mean*mean+sumSquares/count-2*mean*mean)})
                                         (i,math.sqrt(sumSquares/count-mean*mean))})
                             .sortByKey()
                             .map(_._2)
    }
  
    def main(args: Array[String]) {
      
      if (args.length <= 0)
      {
        println("An input libsvm file must be provided")
        return
      }
      
      val file=args(0)
      
      val method= if (args.length>1)
                  {
                    if (args(1).toLowerCase()=="bestfirst")
                      "bestfirst"
                    else
                      "greedystepwise"
                  }
                  else "bestfirst"
                    
      if (method.toLowerCase()!="bestfirst")
        method
      
      //Set up Spark Context
      val conf = new SparkConf().setAppName("PruebaReliefF")//.setMaster("local[8]")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      //conf.set("spark.eventLog.enabled", "true")
      //conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      val sc=new SparkContext(conf)
      
      //Load data from file
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/LargeDatasets/libsvm/isoletTrain.libsvm")
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
      
      val resul=select(sc, data, method)
      
      println("Final selection:")
      println("----------------------------")
      println("Method: "+method)
      println("Merit: "+resul._1)
      println("Set:")
      resul._2.sorted.foreach(println)
    }
  }