package org.apache.spark.mllib.feature

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import java.io.PrintWriter
import java.io.File

object ReliefFFeatureSelector
{
  
    def rankFeatures(sc: SparkContext, data: RDD[LabeledPoint], numNeighbors: Int, attributeNumeric: Array[Boolean], discreteClass: Boolean): RDD[(Int, Double)] =
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
      val numberedData=data.collect()//data.zipWithIndex().map({case x=>(x._2.toInt,x._1)})
      val bnData=sc.broadcast(numberedData)//.collect())
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
    
    def selectDiscrete(indexPairs: RDD[(Int, Int)], bnData: Broadcast[Array[LabeledPoint]], bnTypes: Broadcast[Array[Boolean]], numNeighbors: Int, rangeAttributes:  RDD[(Int, Double)], countsClass: Map[Double, Double], numElems: Double): RDD[(Int, Double)] =
    {
      val dCD=indexPairs
                  .filter({case (x,y) => x<y})
.repartition(256)//Repartition into a suitable number of partitions
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                    /*case (x,y) => val dist=bnData.value(x).features.toArray.zipWithIndex.zip(bnData.value(y).features.toArray)
                                                              .foldLeft(0.0)(
                                                                 {case (sum,((a,i),b)) if (bnTypes.value(i)) => sum+math.abs(a-b) //Numeric
                                                                 case (sum,((a,i),b)) => if (a!=b) sum+1.0 else sum}
                                                                 )
                                  List((x, (y, dist)),(y, (x, dist)))*/
                    case (x,y) => val feat1=bnData.value(x).features.toArray
                                  val feat2=bnData.value(y).features.toArray
                                  var i = 0;
                                  var dist=0.0
                                  // for loop execution with a range
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                       dist=dist+math.abs(feat1(a)-feat2(a))
                                    else
                                      if (feat1(a)!=feat2(a))
                                       dist=dist+1.0
                                  List((x, (y, dist)),(y, (x, dist)))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
.coalesce(128, false)
            .map(//Group by class for each instance so that we can take K neighbors for each class for each instance
                {
                  case (x, nearestNeighborsByClass) => (x, nearestNeighborsByClass.groupBy({case (y, distances) => bnData.value(y).label}))
                })
            .map(//Sort by distance and select K neighbors for each group
                {
                  case(x, nearestNeighborsByClass) =>
                              (x,nearestNeighborsByClass.map(
                                  {
                                    /*case (y, distances) => (y,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                                                                        .take(numNeighbors)) //Take the K nearest neighbors
                                    */
                                    case (y, distances) => val nearest=new Array[(Int,Double)](numNeighbors)
                                                            var curNeighbors=0
                                                            var maxDist=Double.MinValue
                                                            var maxDistIndex=0
                                                            for(a <- distances)
                                                            {
                                                              if (curNeighbors<numNeighbors)
                                                              {
                                                                nearest(curNeighbors)=a
                                                                if (a._2>maxDist)
                                                                {
                                                                  maxDist=a._2
                                                                  maxDistIndex=curNeighbors
                                                                }
                                                                curNeighbors=curNeighbors+1
                                                              }
                                                              else
                                                                if (a._2<maxDist)
                                                                {
                                                                  nearest(maxDistIndex)=a
                                                                  maxDist=a._2
                                                                  for(n <- 0 until nearest.length)
                                                                  {
                                                                    if (nearest(n)._2>maxDist)
                                                                    {
                                                                      maxDist=nearest(n)._2
                                                                      maxDistIndex=n
                                                                    }
                                                                  } 
                                                                }
                                                            }
                                                            val nearestRet=new Array[(Int,Double)](curNeighbors)
                                                            for(i <- 0 until curNeighbors)
                                                              nearestRet(i)=(nearest(i)._1,curNeighbors)
                                                            (y, nearestRet)
                                   })
                                   //.map({case(y,distances) => (y,distances.map({case(y,d) => (y,distances.length)}))}) //Add the number of neighbors so that we can divide later
                              )
                })
            .flatMap(//Ungroup everything in order to get closer to having addends
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2)}).flatten.map({y => (x,y)})
                })
            .map(//Compute multipliers for each addend depending on their class
                {
                  case(x,(y,k)) if (bnData.value(x).label==bnData.value(y).label) => (x, y, -1.0/k)
                  case(x,(y,k)) if (bnData.value(x).label!=bnData.value(y).label) => (x, y, countsClass.get(bnData.value(y).label).get/((1.0-countsClass.get(bnData.value(x).label).get)*k))
                }
                )
            .flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  /*case(x, y, s) => bnData.value(x).features.toArray.zip(bnData.value(y).features.toArray).zipWithIndex.map(
                                                                             {case ((x,y),i) if (bnTypes.value(i)) => (i, math.abs(x-y)*s)//Numeric
                                                                             case ((fx,fy),i) => (i, if (fx!=fy) 1.0 else 0.0)})//Nominal
                  */
                  case(x, y, s) => val feat1=bnData.value(x).features.toArray
                                  val feat2=bnData.value(y).features.toArray
                                  var i = 0;
                                  var res:Array[(Int, Double)] = new Array[(Int, Double)](feat1.length)
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                       res(a)=(a,math.abs(feat1(a)-feat2(a))*s)
                                    else
                                      if (feat1(a)!=feat2(a))
                                       res(a)=(a,1.0)
                                      else
                                       res(a)=(a,0.0)
                                  res
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
    
    def selectNumeric(indexPairs: RDD[(Int, Int)], bnData: Broadcast[Array[LabeledPoint]], bnTypes: Broadcast[Array[Boolean]], numNeighbors: Int, rangeAttributes:  RDD[(Int, Double)], countsClass: Map[Double, Double], numElems: Double, rangeClass: Double): RDD[(Int, Double)] =
    {
      val dCD=indexPairs //Will compare each instance with every other
                  //.repartition(8)//Repartition into a suitable number of partitions
                  .filter({case (x,y) => x<y})
                  .flatMap(//Remove comparisons between an instance and itself and compute distances
                  {
                    /*case (x,y) => val dist=bnData.value(x).features.toArray.zipWithIndex.zip(bnData.value(y).features.toArray)
                                                              .foldLeft(0.0)(
                                                                 {case (sum,((a,i),b)) if (bnTypes.value(i)) => sum+math.abs(a-b) //Numeric
                                                                 case (sum,((a,i),b)) => if (a!=b) sum+1.0 else sum}
                                                                 )*/
                    case (x,y) => val feat1=bnData.value(x).features.toArray
                                  val feat2=bnData.value(y).features.toArray
                                  var i = 0;
                                  var dist=0.0
                                  // for loop execution with a range
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                       dist=dist+math.abs(feat1(a)-feat2(a))
                                    else
                                      if (feat1(a)!=feat2(a))
                                       dist=dist+1.0
                                  List((x, (y, dist)),(y, (x, dist)))
                  })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            .map(//Sort by distance and select K neighbors for each instance
                {
                  /*case(x, distances) =>
                              (x,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                          .take(numNeighbors)) //Take the K nearest neighbors
                  */
                  case (y, distances) => val nearest=new Array[(Int,Double)](numNeighbors)
                                          var curNeighbors=0
                                          var maxDist=Double.MinValue
                                          var maxDistIndex=0
                                          for(a <- distances)
                                          {
                                            if (curNeighbors<numNeighbors)
                                            {
                                              nearest(curNeighbors)=a
                                              if (a._2>maxDist)
                                              {
                                                maxDist=a._2
                                                maxDistIndex=curNeighbors
                                              }
                                              curNeighbors=curNeighbors+1
                                            }
                                            else
                                              if (a._2<maxDist)
                                              {
                                                nearest(maxDistIndex)=a
                                                maxDist=a._2
                                                for(n <- 0 until nearest.length)
                                                {
                                                  if (nearest(n)._2>maxDist)
                                                  {
                                                    maxDist=nearest(n)._2
                                                    maxDistIndex=n
                                                  }
                                                }
                                              }
                                          }
                                          val nearestRet=new Array[(Int,Double)](numNeighbors)
                                          for(i <- 0 until curNeighbors)
                                            nearestRet(i)=(nearest(i)._1,curNeighbors)
                                          (y, nearestRet)
                })
            /*.map(
                {
                  case(x,distances) =>
                              (x,distances.map({case(y,d) => (y,distances.length)})) //Add the number of neighbors so that we can divide later
                })*/
            .flatMap(//Ungroup everything in order to closer to addends
                {
                  case (x, distances) =>
                      distances.flatMap({y => List((x,y))})
                })
      dCD.cache()          
      val m_ndc=dCD.map(
                {
                  case (x, (y,k)) => (math.abs(bnData.value(x).label-bnData.value(y).label))
                }) //Will be normalized when computing the weight
                .reduce(_+_)
      
      val numNeighborsObtained=dCD.first()._2._2 //Check the number of neighbors computed.

      val weights=dCD.flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  /*case(x, (y, k)) => bnData.value(x).features.toArray.zip(bnData.value(y).features.toArray).zipWithIndex.map(
                                                                             {
                                                                               case ((a,b),i) if (bnTypes.value(i)) => (i,(math.abs(a-b), math.abs(a-b)*(math.abs(bnData.value(x).label-bnData.value(y).label))))//Numeric
                                                                               case ((fx,fy),i) => (i, (if (fx!=fy) 1.0 else 0.0, if (fx!=fy) math.abs(bnData.value(x).label-bnData.value(y).label) else 0.0))})//Nominal
                  */
                  case(x, (y, s)) => val feat1=bnData.value(x).features.toArray
                                  val feat2=bnData.value(y).features.toArray
                                  var i = 0;
                                  var res:Array[(Int, (Double,Double))] = new Array[(Int, (Double,Double))](feat1.length)
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                       res(a)=(a,(math.abs(feat1(a)-feat2(a)), math.abs(feat1(a)-feat2(a))*math.abs(bnData.value(x).label-bnData.value(y).label)))
                                    else
                                      if (feat1(a)!=feat2(a))
                                       res(a)=(a,(1.0,math.abs(bnData.value(x).label-bnData.value(y).label)))
                                      else
                                       res(a)=(a,(0.0,0.0))
                                  res
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
      
      var fileOut=file.substring(0,file.lastIndexOf("."))+"-out.txt"
      
      //Set up Spark Context
      val conf = new SparkConf().setAppName("PruebaReliefF")//.setMaster("local[8]")
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.eventLog.enabled", "true")
//      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      
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
                          args(2).toCharArray().map({case c => c=='N'})
                         else
                          (0 to data.first().features.size-1).map({case c => true}).toArray
      if (attributeTypes.length!=data.first().features.size)
      {
        println("The number of features does not match the instances in the file")
        return
      }
      
      val discreteClass=(args.length<4) || ((args(3)!="n") && (args(3)!="N"))
      
      val pw = new PrintWriter(new File(fileOut))
      
      println("File: "+file)
      pw.println("File: "+file)
      println("Number of neighbors: "+numNeighbors)
      pw.println("Number of neighbors: "+numNeighbors)
      print("Attribute types: ")
      pw.print("Attribute types: ")
      attributeTypes.foreach { x => print(if (x) "N" else "D") }
      println
      pw.println
      println("Class: "+(if (discreteClass) "Discrete" else "Numeric"))
      pw.println("Class: "+(if (discreteClass) "Discrete" else "Numeric"))
      
      val startTime=System.currentTimeMillis()
      
      //Select features
      val features=rankFeatures(sc, data, numNeighbors, attributeTypes, discreteClass)
      //Print results
      features.sortBy(_._2, false).collect().foreach({case (index, weight) =>
                                                              printf("Attribute %d: %f\n",index,weight)
                                                              pw.println("Attribute "+"%d".format(index)+": "+"%f".format(weight))})
      
      println("Computed in "+(System.currentTimeMillis()-startTime)+" milliseconds")
      pw.println("Computed in "+(System.currentTimeMillis()-startTime)+" milliseconds")
      
      pw.close
      
      //Stop the Spark Context
      sc.stop()
    }
  }