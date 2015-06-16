package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

object ReliefFFeatureSelector
{
  
    def selectFeatures(sc: SparkContext, data: RDD[LabeledPoint], numNeighbors: Int, attributeNumeric: Map[Int, Boolean], discreteClass: Boolean): RDD[(Int, Double)] =
    {
      data.cache()
      //Count total number of instances
      val numElems=data.count().toDouble
      printf("NumElems: %f\n\n",numElems)
      
      //Compute probability for each class
      val countsClass=data.map(_.label)
                          .countByValue
                          .map(
          {
            case (value, count) => (value,count/numElems)
          })
      //printf("Class probs:\n-----------------\n")
      //countsClass.toArray.sortBy(_._1).foreach({x => printf(" %f\t%f\n",x._1,x._2)})
      
      //Compute range of each attribute to normalize distances
      val rangeAttributes=data.flatMap(_.features.toArray.zipWithIndex) //Separate in (numAttribute,value) pairs
                            .map({case (value, index) => (index, (value, value))}) //Rearrange to compute max and min by reducing
                            .reduceByKey(//Compute max and min for each attribute index
                                {
                                  case ((max1, min1), (max2, min2)) =>
                                    (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                                })
                            .map(//Calculate range for each attribute
                                {
                                  case (index, (max, min)) => (index, max-min)//Numeric
                                  //case (index, (max, min)) => (index, 1) //Nominal
                                })
      //printf("\n\nRange attributes:\n-----------------\n")
      //rangeAttributes.foreach(println)
      
      //Data is broadcasted in order to reduce memory usage. Indices are used to access its elements.
      val numberedData=data.zipWithIndex().map({case x=>(x._2.toInt,x._1)})
      val bnData=sc.broadcast(numberedData.collect().toMap)
      val bnTypes=sc.broadcast(attributeNumeric)
      var indices=sc.parallelize(0 to numElems.toInt-1)
      var cart=indices.cartesian(indices)//Will compare each instance with every other
      
      if (discreteClass)
      {
        data.unpersist(false)
        return selectDiscrete(cart, bnData, bnTypes, numNeighbors, rangeAttributes, countsClass.toMap, numElems);
      }
      
      val maxMinClass=data.map(
                            {case x => (x.label, x.label)}
                            )
                         .reduce(//Compute max and min
                            {
                              case ((max1, min1), (max2, min2)) =>
                                (if (max1>max2) max1 else max2, if (min1<min2) min1 else min2)
                            })
      val rangeClass=maxMinClass._1-maxMinClass._2
      //printf("\n\nRange Class:"+rangeClass+"\n")
      data.unpersist(false)
      return selectNumeric(cart, bnData, bnTypes, numNeighbors, rangeAttributes, countsClass.toMap, numElems, rangeClass)
    }
    
    def selectDiscrete(indexPairs: RDD[(Int, Int)], bnData: Broadcast[Map[Int, LabeledPoint]], bnTypes: Broadcast[Map[Int, Boolean]], numNeighbors: Int, rangeAttributes:  RDD[(Int, Double)], countsClass: Map[Double, Double], numElems: Double): RDD[(Int, Double)] =
    {
      val dCD=indexPairs
                  //.repartition(8)//Repartition into a suitable number of partitions
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                     case (x,y) if (x==y) => None
                     case (x,y) if (x!=y) => Some(x, (y, bnData.value.get(x).get.features.toArray.zipWithIndex.zip(bnData.value.get(y).get.features.toArray).map(
                                                                                     {case ((a,i),b) if (bnTypes.value.get(i).get) => math.abs(a-b) //Numeric
                                                                                     case ((a,i),b) => if (a!=b) 1.0 else 0.0} //Nominal
                                                                                     ).sum))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            .map(//Group by class for each instance so that we can take K neighbors for each class for each instance
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (y, distances) => bnData.value.get(y).get.label}))
                })
            .map(//Sort by distance and select K neighbors for each group
                {
                  case(x, nearestNeighborsByClass) =>
                              (x,nearestNeighborsByClass.map(
                                  {
                                    case (y, distances) => (y,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                                                                        .take(numNeighbors)) //Take the K nearest neighbors
                                   })
                                   .map({case(y,distances) => (y,distances.map({case(y,d) => (y,distances.length)}))}) //Add the number of neighbors so that we can divide later
                              )
                })
            .flatMap(//Ungroup everything in order to closer to addends
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2)}).flatten.map({y => (x,y)})
                })
            .map(//Compute multipliers for each addend depending on their class
                {
                  case(x,(y,k)) if (bnData.value.get(x).get.label==bnData.value.get(y).get.label) => (x, y, -1.0/k)
                  case(x,(y,k)) if (bnData.value.get(x).get.label!=bnData.value.get(y).get.label) => (x, y, countsClass.get(bnData.value.get(y).get.label).get/((1.0-countsClass.get(bnData.value.get(x).get.label).get)*k))
                }
                )
            .flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  case(x, y, s) => bnData.value.get(x).get.features.toArray.zip(bnData.value.get(y).get.features.toArray).zipWithIndex.map(
                                                                             {case ((x,y),i) if (bnTypes.value.get(i).get) => (i, math.abs(x-y)*s)//Numeric
                                                                             case ((fx,fy),i) => (i, if (fx!=fy) 1.0 else 0.0)})//Nominal
                }
                )
            .reduceByKey({_ + _})//Sum up for each attribute
            .join(rangeAttributes)//In order to divide by the range of each attribute
            .map(//Add 1 to the attribNum so that is in the [1,N] range and divide each result by m and k.
                {
                  case(attribNum, (sum, range)) => (attribNum+1, sum/(range*numElems))
                })
      return dCD;
    }
    
    def selectNumeric(indexPairs: RDD[(Int, Int)], bnData: Broadcast[Map[Int, LabeledPoint]], bnTypes: Broadcast[Map[Int, Boolean]], numNeighbors: Int, rangeAttributes:  RDD[(Int, Double)], countsClass: Map[Double, Double], numElems: Double, rangeClass: Double): RDD[(Int, Double)] =
    {
      val dCD=indexPairs //Will compare each instance with every other
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                     case (x,y) if (x==y) => None
                     case (x,y) if (x!=y) => Some(x, (y, bnData.value.get(x).get.features.toArray.zipWithIndex.zip(bnData.value.get(y).get.features.toArray).map(
                                                                                     {case ((a,i),b) if (bnTypes.value.get(i).get) => math.abs(a-b) //Numeric
                                                                                     case ((a,i),b) => if (a!=b) 1.0 else 0.0}
                                                                                     ).sum))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            .map(//Sort by distance and select K neighbors for each instance
                {
                  case(x, distances) =>
                              (x,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                          .take(numNeighbors)) //Take the K nearest neighbors
                })
            .map(
                {
                  case(x,distances) =>
                              (x,distances.map({case(y,d) => (y,distances.length)})) //Add the number of neighbors so that we can divide later
                })
            .flatMap(//Ungroup everything in order to closer to addends
                {
                  case (x, distances) =>
                      distances.flatMap({y => List((x,y))})
                })
      dCD.cache()          
      val m_ndc=dCD.map(
                {
                  case (x, (y,k)) => (math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))
                }) //Will be normalized when computing the weight
                .reduce(_+_)
      
      val numNeighborsObtained=dCD.first()._2._2 //Check the number of neighbors computed.

      val weights=dCD.flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  case(x, (y, k)) => bnData.value.get(x).get.features.toArray.zip(bnData.value.get(y).get.features.toArray).zipWithIndex.map(
                                                                             {
                                                                               case ((a,b),i) if (bnTypes.value.get(i).get) => (i,(math.abs(a-b), math.abs(a-b)*(math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label))))//Numeric
                                                                               case ((fx,fy),i) => (i, (if (fx!=fy) 1.0 else 0.0, if (fx!=fy) math.abs(bnData.value.get(x).get.label-bnData.value.get(y).get.label) else 0.0))})//Nominal
                })
            .reduceByKey({case ((m_nda1, m_ndcda1), (m_nda2, m_ndcda2)) => (m_nda1 + m_nda2, m_ndcda1 + m_ndcda2)})//Sum up for each attribute
            .join(rangeAttributes)//In order to divide by the range of each attribute
            .map(//Add 1 to the attribNum so that is in the [1,N] range and compute weights.
                {
                  case(attribNum, ((m_nda, m_ndcda),range)) => (attribNum+1, (m_ndcda/m_ndc - ((rangeClass*m_nda - m_ndcda)/(numNeighborsObtained*rangeClass*numElems-m_ndc)))/range)
                })
                
      dCD.unpersist(false)
      
      return weights
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
      conf.set("spark.eventLog.enabled", "true")
      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/sparklog")
      val sc=new SparkContext(conf)
      
      //Load data from file
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/LargeDatasets/libsvm/isoletTrain.libsvm")
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      
      //Set maximum number of near neighbors to be taken into account for each instance
      val numNeighbors=if (args.length>=2)
                        args(1).toInt
                       else
                        10

      //Set the type (numeric/discrete) for each attribute and class
      val attributeTypes=if (args.length>=3)
                          args(2).toCharArray().zipWithIndex.map({case (c,i) => (i, c=='N')}).toSeq
                         else
                          (0 to data.first().features.size).map((_,true))
      if (attributeTypes.length!=data.first().features.size)
      {
        println("The number of features does not match the instances in the file")
        return
      }
      val discreteClass=(args.length<4) || (args(3)!="n")
      
      println("File: "+file)
      println("Number of neighbors: "+numNeighbors)
      print("Attribute types: ")
      attributeTypes.foreach { x => print(if (x._2) "N" else "D") }
      println
      println("Class: "+(if (discreteClass) "Discrete" else "Numeric"))
      
      //Select features
      val features=selectFeatures(sc, data, numNeighbors, attributeTypes.toMap, discreteClass)
      
      //Print results
      features.sortBy(_._2, false).collect().foreach({case (index, weight) => printf("Attribute %d: %f\n",index,weight)})
      
      //Stop the Spark Context
      sc.stop()
    }
  }