package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.mllib.classification.SVMWithSGD
import scala.collection.mutable.Stack
import scala.util.control.Breaks._

object SVMRFEFeatureSelector
{
    def rankFeatures(sc: SparkContext, data: RDD[LabeledPoint]): Array[(Int, Double)] =
    {
      data.cache()
      //Count total number of instances
      val numElems=data.count().toDouble
      printf("NumElems: %f\n\n",numElems)

      val ranking=new Stack[Int]
      var dataStep=data
      
      val model = SVMWithSGD.train(data, 100)//100 iterations
      
      model.weights.toArray.zipWithIndex.foreach(println)
      
      //Take the STEP attributes with a smaller weight, add them to the ranking and remove them from the data set
      var STEP=3
      if (STEP>model.weights.size)
        STEP=model.weights.size
      val mins=new Array[(Int,Double)](STEP)
      var maxIndex=0
      var maxValue=Double.MaxValue
      var numMins=0
      for (a <- 0 until model.weights.size)
      {
        var w=model.weights(a)
        w=w*w
        if (numMins<STEP)
        {
          mins(numMins)=(a,w)
          if (w>maxValue)
          {
            maxIndex=numMins
            maxValue=w
          }
          numMins=numMins+1
        }
        else
        {
          if (w<maxValue)
          {
            mins(maxIndex)=(a,w)
            maxValue=w
            for(n <- 0 until mins.length)
            {
              if (mins(n)._2>maxValue)
              {
                maxValue=mins(n)._2
                maxIndex=n
              }
            }
          }
        }
      }
      
      //Add attributes to ranking
      val sortedMins=mins.sortBy({case (index, weight) => -weight})
      for (a <- 0 until sortedMins.length)
        ranking.push(sortedMins(a)._1)
        
      val sortedIndices=mins.sortBy({case (index, weight) => index})
        
      //Remove attributes from dataset
      dataStep=dataStep.map({ case x =>
                        val newFeatures=new Array[Double](x.features.size-STEP)
                        var newIndex=0
                        var indexSorted=0
                        for (i <- 0 until x.features.size)
                        {
                          if (i == sortedIndices(indexSorted)._1)
                          {
                            indexSorted=indexSorted+1
                            if (indexSorted>=sortedIndices.length)
                              break
                          }
                          else
                          {
                            newFeatures(newIndex)=x.features.size
                            newIndex=newIndex+1
                          }
                        }
                        new LabeledPoint(x.label, newFeatures)})
      //Así quedan los índices mal. ¿Se puede aprovechar sparsity para quitar valores?
      //UNPERSIST!!
      
      ranking.foreach(println)
      
      return (0 to 10).toArray.map({case x => (x,x*1.0)})
    }
    
    def main(args: Array[String])
    {
      if (args.length <= 0)
      {
        println("An input libsvm file must be provided")
        return
      }
      
      var file=args(0)
      
      //Set up Spark Context
      val conf = new SparkConf().setAppName("PruebaReliefF").setMaster("local[8]")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.eventLog.enabled", "true")
//      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      val sc=new SparkContext(conf)
      
      //Load data from file
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/LargeDatasets/libsvm/isoletTrain.libsvm")
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      
      println("File: "+file)
      
      //Select features
      val features=rankFeatures(sc, data)
      //Print results
      //features.sortBy(_._2, false).collect().foreach({case (index, weight) => printf("Attribute %d: %f\n",index,weight)})
      
      //Stop the Spark Context
      sc.stop()
    }
  }