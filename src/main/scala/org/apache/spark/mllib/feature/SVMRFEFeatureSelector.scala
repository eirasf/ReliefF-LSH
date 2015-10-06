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
import breeze.linalg.Vector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.optimization.L1Updater


object SVMRFEFeatureSelector
{
    def rankFeatures(sc: SparkContext, data: RDD[LabeledPoint], numberOfAttributes:Int): Array[Int] =
    {
      val ranking=new Array[Int](numberOfAttributes)
      
      data.cache()
      //Count total number of instances
      val numElems=data.count().toDouble
      printf("NumElems: %f\n\n",numElems)
      
      val classes=data.map({ case x => x.label }).distinct().collect()
      val partialRankings=new Array[Stack[Int]](classes.length)
      for(i <- (0 until classes.length))
      {
        partialRankings(i)=rankFeaturesSingleClass(sc, data.map({case x if x.label==classes(i) => new LabeledPoint(1.0, x.features)
                                                                  case x => new LabeledPoint(0.0, x.features)}), numberOfAttributes)
        print("Class "+i+": ")
        partialRankings(i).foreach({case x => print(x+",")})
        println()
      }
      
                                                                  
      //Componer ranking
      var i=0
      var lastClass=0
      while(i<ranking.length)
      {
        while((i<ranking.length) && (lastClass<classes.length))
        {
          val r=partialRankings(lastClass).pop()
          if (!ranking.contains(r))
          {
            ranking(i)=r
            i=i+1
          }
          lastClass=lastClass+1
        }
        lastClass=lastClass%classes.length
      }
      
      return ranking
    }
    
    def rankFeaturesSingleClass(sc: SparkContext, data: RDD[LabeledPoint], numberOfAttributes:Int): Stack[Int] =
    {
      val ranking=new Stack[Int]
      var dataStep=data
      println("Left: "+dataStep.first().features.size)
      
      var origIndices=(1 to data.first().features.size).toArray 
      
      while(ranking.length<numberOfAttributes)
      {
        //dataStep.take(14)(13).features.foreachActive({case x => println(x._1+" - "+x._2)})
        println("Training....")
        
        dataStep.cache()
        //dataStep.foreach { x => println(x.features) }
        //Usar L1
       /* val svmAlg = new SVMWithSGD()
        svmAlg.optimizer.
          setNumIterations(100).
          setRegParam(0.02).
          setUpdater(new L1Updater)
        //val modelL1 = svmAlg.run(training)
          val model = svmAlg.run(dataStep)*/
        val model = SVMWithSGD.train(dataStep, 100)//100 iterations
        dataStep.unpersist()
        
        //Take the STEP attributes with a smaller weight, add them to the ranking and remove them from the data set
        //var STEP=numberOfAttributes
        var STEP=1
        if (STEP>model.weights.size)
          STEP=model.weights.size
        if (STEP>numberOfAttributes-ranking.length)
          STEP=numberOfAttributes-ranking.length
        val mins=new Array[(Int,Double)](STEP)
        var maxIndex=0
        var maxValue=Double.MaxValue
        var numMins=0
        
        //DEBUG - Print weights
        //model.weights.foreachActive({case x =>println("w["+x._1+"]="+x._2)})
        model.weights.foreachActive(
                { case a =>
                  var w=a._2
                  w=w*w
                  if (numMins<STEP)
                  {
                    mins(numMins)=(a._1,w)
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
                      mins(maxIndex)=(a._1,w)
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
                })
        //mins.foreach({case x =>println("m["+x._1+"]="+x._2)})
        //Add attributes to ranking
        val sortedMins=mins.sortBy({case (index, weight) => -weight})
        for (a <- 0 until sortedMins.length)
          ranking.push(origIndices(sortedMins(a)._1))
          
        val sortedIndices=mins.sortBy({case (index, weight) => index})
        
        val dense=dataStep.first().features.isInstanceOf[DenseVector]
        
        //println("Quitar:")
        //sortedIndices.foreach(println)
        
        if (ranking.length<numberOfAttributes)
        {
          //Remove attributes from dataset
          dataStep=dataStep.map({ case x =>
                                    var newSize=x.features.size-STEP
                                    if (x.features.isInstanceOf[SparseVector])
                                      newSize=x.features.asInstanceOf[SparseVector].indices.length
                                    var newValues=new Array[Double](newSize)
                                    var newIndices=new Array[Int](newSize)
                                    var curIndex=0
                                    var indexSorted=0
                                    var removed=0
                                    var skipped=0
                                    x.features.foreachActive({case x =>
                                          //println("Checking: "+x._1+","+x._2)
                                          if ((indexSorted<STEP) && (x._1 == sortedIndices(indexSorted)._1))
                                          {
                                            indexSorted=indexSorted+1
                                            removed=removed+1
                                            skipped=skipped+1
                                            //println("---------Saltado "+x._1)
                                          }
                                          else
                                          {
                                            while ((indexSorted<STEP) && (x._1 >= sortedIndices(indexSorted)._1))
                                            {
                                              indexSorted=indexSorted+1
                                              skipped=skipped+1
                                            }
                                            newValues(curIndex)=x._2
                                            newIndices(curIndex)=x._1-skipped
                                            curIndex=curIndex+1
                                          }
                                      })
                                    //newValues.foreach(println)
                                    //println("T:"+(x.features.size-STEP)+" I:"+newIndices.length+" V:"+newValues.length+" C:"+curIndex)
                                    if (removed>0)
                                    {
                                      newIndices=newIndices.dropRight(removed)
                                      newValues=newValues.dropRight(removed)
                                    }
                                    new LabeledPoint(x.label, new SparseVector(x.features.size-STEP, newIndices, newValues))
                              })
        }
        
        val temp = new Array[Int](dataStep.first().features.size) //New indices
        var indexSorted=0
        var k=0
        for (i <- 0 until origIndices.length)
        {
          if ((indexSorted<STEP) && (i == sortedIndices(indexSorted)._1))
          {
            indexSorted=indexSorted+1
          }
          else
          {
            while ((indexSorted<STEP) && (i > sortedIndices(indexSorted)._1-1))
            {
              indexSorted=indexSorted+1
            }
            temp(k)=origIndices(i)
            k=k+1
          }
        }
        origIndices=temp
        
        //println("Índices")
        //origIndices.foreach(println)
        
        println(ranking.length+"/"+numberOfAttributes)
        //println("Left: "+dataStep.first().features.size)
      }
      //UNPERSIST!!
      return ranking
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
      
      println("File: "+file)
      //val normalizer1 = new Normalizer()
      //val data1 = data.map(x => new LabeledPoint(x.label, normalizer1.transform(x.features)))
      
      
      //Select features
      val features=rankFeatures(sc, data, 28)//data.first().features.size)
      //Print results
      features.foreach(println)
      //features.sortBy(_._2, false).collect().foreach({case (index, weight) => printf("Attribute %d: %f\n",index,weight)})
      
      //Stop the Spark Context
      sc.stop()
    }
  }