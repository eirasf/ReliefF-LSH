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
import es.udc.graph.LSHLookupKNNGraphBuilder
import es.udc.graph.LSHKNNGraphBuilder
import es.udc.graph.DummyGroupingProvider
import es.udc.graph.DistanceProvider
import es.udc.graph.KNiNeConfiguration
import es.udc.graph.GroupingProvider
import es.udc.graph.EuclideanLSHasher
import es.udc.graph.BroadcastLookupProvider
import es.udc.graph.LookupProvider

class ReliefFDistanceProvider(bnTypes: Broadcast[Array[Boolean]], normalizingDict: Broadcast[scala.collection.Map[Int, Double]]) extends DistanceProvider
{
  def getDistance(p1:LabeledPoint,p2:LabeledPoint):Double=
  {
    val feat1=p1.features.toArray
    val feat2=p2.features.toArray
    var i = 0;
    var dist=0.0
    // for loop execution with a range
    for( a <- 0 to feat1.length-1)
      if (bnTypes.value(a))
      {
         val range=normalizingDict.value(a)
         dist=dist+math.abs(feat1(a)-feat2(a))/range
      }
      else
        if (feat1(a)!=feat2(a))
         dist=dist+1.0
    return dist
  }
}

class ReliefFGroupingProvider(classNames:Iterable[Double]) extends GroupingProvider
{
  private def classMap=classNames.zipWithIndex.toMap
  def numGroups=classMap.size
  def getGroupId(p1:LabeledPoint):Int=
  {
    return classMap.get(p1.label).get
  }
}

object ReliefFFeatureSelector
{
    val DEFAULT_METHOD="lsh"
    val DEFAULT_K=10
    private def getNNearest(distances:Iterable[(Long, Double)], numberOfSelected:Int):List[(Long,Double)]=
    {
      val nearest=new Array[(Long,Double)](numberOfSelected)
      var curNeighbors=0
      var maxDist=Double.MinValue
      var maxDistIndex=0
      for(a <- distances)
      {
        if (curNeighbors<numberOfSelected)
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
      return nearest.toList
      //val nearestRet=new Array[(Long,Double)](curNeighbors)
      //for(i <- 0 until curNeighbors)
      //  nearestRet(i)=(nearest(i)._1,curNeighbors)
      //return nearestRet.toList
    }
    
    def getKNNGraph(sc: SparkContext, data:RDD[(LabeledPoint,Long)], numNeighbors:Int, bnTypes: Broadcast[Array[Boolean]], normalizingDict: Broadcast[scala.collection.Map[Int, Double]]):(RDD[(Long,List[(Long,Double)])],LookupProvider)=
    {
      val (graph,lookup)=getGroupedKNNGraph(sc, data, numNeighbors, bnTypes, normalizingDict, new DummyGroupingProvider())
      return (graph.map({case (x,groupedNeighbors) => (x,groupedNeighbors.head._2)}),lookup)
    }
    
    def getGroupedKNNGraph(sc: SparkContext, data:RDD[(LabeledPoint,Long)], numNeighbors:Int, bnTypes: Broadcast[Array[Boolean]], normalizingDict: Broadcast[scala.collection.Map[Int, Double]], grouper:GroupingProvider):(RDD[(Long,List[(Int,List[(Long,Double)])])],LookupProvider)=
    {
      var rddIndices=data.map(_._2)
      var indexPairs=rddIndices.cartesian(rddIndices)
      val lookup=new BroadcastLookupProvider(data.map(_.swap))
      val measurer=new ReliefFDistanceProvider(bnTypes, normalizingDict)
      val kNNGraph=indexPairs //Will compare each instance with every other
           //.repartition(8)//Repartition into a suitable number of partitions
            .filter({case (x,y) => x<y})
            .flatMap(//Remove comparisons between an instance and itself and compute distances
            {
              /*case (x,y) => val dist=bnData.value(x).features.toArray.zipWithIndex.zip(bnData.value(y).features.toArray)
                                                        .foldLeft(0.0)(
                                                           {case (sum,((a,i),b)) if (bnTypes.value(i)) => sum+math.abs(a-b) //Numeric
                                                           case (sum,((a,i),b)) => if (a!=b) sum+1.0 else sum}
                                                           )*/
              case (x,y) => val dist=measurer.getDistance(lookup.lookup(x), lookup.lookup(y))
                            List((x, (y, dist)),(y, (x, dist)))
            })//.filter(_!=null)//By using flatMap and None/Some values this filter is avoided
            .groupByKey//Group by instance
            .map(//Group by class for each instance so that we can take K neighbors for each class for each instance
                {
                  case (x, neighbors) => (x, neighbors.groupBy(
                                              {
                                                case (y, distances) =>
                                                  val yVal=lookup.lookup(y)
                                                  grouper.getGroupId(yVal)
                                              }).toList
                                         )
                })
            .map(//Sort by distance and select K neighbors for each group
                {
                  case(x, neighborsByClass) =>
                              (x,neighborsByClass.map(
                                  {
                                    /*case (y, distances) => (y,distances.toSeq.sortBy({case(y,d) => d}) //Sort by distance
                                                                                        .take(numNeighbors)) //Take the K nearest neighbors
                                    */
                                    case (cl, distances) => (cl, getNNearest(distances, numNeighbors))
                                   })
                                   //.map({case(y,distances) => (y,distances.map({case(y,d) => (y,distances.length)}))}) //Add the number of neighbors so that we can divide later
                              )
                })
      return (kNNGraph, lookup)
    }
    
    def getKNNGraphFromKNiNe(sc: SparkContext, data:RDD[(Long,LabeledPoint)], numNeighbors:Int, bnTypes: Broadcast[Array[Boolean]], normalizingDict: Broadcast[scala.collection.Map[Int, Double]], lshConf:KNiNeConfiguration):(RDD[(Long,List[(Long,Double)])],LookupProvider)=
    {
      //val (graph,lookup)=BruteForceKNNGraphBuilder.parallelComputeGraph(data, numNeighbors, new ReliefFDistanceProvider(bnTypes, normalizingDict))
      val builder=new LSHLookupKNNGraphBuilder(data)
      val graph=builder.computeGraph(data, numNeighbors, lshConf.keyLength.get, lshConf.numTables.get, lshConf.radius0, lshConf.maxComparisons, new ReliefFDistanceProvider(bnTypes, normalizingDict))
      return (graph,
              builder.lookup)
    }
    
    def getGroupedKNNGraphFromKNiNe(sc: SparkContext, data:RDD[(Long,LabeledPoint)], numNeighbors:Int, bnTypes: Broadcast[Array[Boolean]], normalizingDict: Broadcast[scala.collection.Map[Int, Double]], grouper:GroupingProvider, lshConf:KNiNeConfiguration):(RDD[(Long,List[(Int,List[(Long,Double)])])],LookupProvider)=
    {
      //val (graph,lookup)=BruteForceKNNGraphBuilder.parallelComputeGraph(data, numNeighbors, new ReliefFDistanceProvider(bnTypes, normalizingDict))
      val builder=new LSHLookupKNNGraphBuilder(data)
      val graph=if (lshConf.keyLength.isDefined && lshConf.numTables.isDefined)
                  builder.computeGroupedGraph(data, numNeighbors, lshConf.keyLength.get, lshConf.numTables.get, lshConf.radius0, lshConf.maxComparisons, new ReliefFDistanceProvider(bnTypes, normalizingDict), grouper)
                else
                {
                  val factor=4.0
                  val (hasher,nComps,suggestedRadius)=EuclideanLSHasher.getHasherForDataset(data, (factor*numNeighbors).toInt)
                  builder.computeGroupedGraph(data, numNeighbors, hasher, lshConf.radius0, lshConf.maxComparisons, new ReliefFDistanceProvider(bnTypes, normalizingDict), grouper)
                }
      return (graph,builder.lookup)
    }
    
    def rankFeatures(sc: SparkContext, data: RDD[LabeledPoint], numNeighbors: Int, attributeNumeric: Array[Boolean], discreteClass: Boolean, lshConf:Option[KNiNeConfiguration]): RDD[(Int, Double)] =
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
      
      
      val numberedData=data.zipWithIndex()
      printf("MaxIndex: %d\n\n",numberedData.map(_._2).max())
      val bnTypes=sc.broadcast(attributeNumeric)
      
      val normalizingDict=rangeAttributes.collectAsMap()
      val bnNormalizingDict=sc.broadcast(normalizingDict)
      
      if (discreteClass)
      {
        data.unpersist(false)
        return selectDiscrete(sc, numberedData, bnTypes, numNeighbors, bnNormalizingDict, countsClass.toMap, numElems, lshConf);
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
      return selectNumeric(sc, numberedData, bnTypes, numNeighbors, bnNormalizingDict, countsClass.toMap, numElems, rangeClass, lshConf)
    }
    
    def selectDiscrete(sc: SparkContext, data: RDD[(LabeledPoint,Long)], bnTypes: Broadcast[Array[Boolean]], numNeighbors: Int, normalizingDict: Broadcast[scala.collection.Map[Int, Double]], countsClass: Map[Double, Double], numElems: Double, lshConf:Option[KNiNeConfiguration]): RDD[(Int, Double)] =
    {
      val (kNNGraph,lookup)=if (lshConf.isDefined)
                              getGroupedKNNGraphFromKNiNe(sc, data.map(_.swap), numNeighbors, bnTypes, normalizingDict, new ReliefFGroupingProvider(countsClass.keys), lshConf.get) 
                            else
                              getGroupedKNNGraph(sc, data, numNeighbors, bnTypes, normalizingDict, new ReliefFGroupingProvider(countsClass.keys))
      val dCD=kNNGraph
              .flatMap(//Ungroup everything in order to get closer to having addends
                {
                  case (x, nearestNeighborsByClass) =>
                    nearestNeighborsByClass.flatMap({y=>List(y._2.map({case (p,d) => (p,y._2.length)}))}).flatten.map({case (otherElement,k) => (x,(otherElement,k))})
                })
            .map(//Compute multipliers for each addend depending on their class
                {
                  case(x,(y,k)) if (lookup.lookup(x).label==lookup.lookup(y).label) => (x, y, -1.0/k)
                  case(x,(y,k)) if (lookup.lookup(x).label!=lookup.lookup(y).label) => (x, y, countsClass.get(lookup.lookup(y).label).get/((1.0-countsClass.get(lookup.lookup(x).label).get)*k))
                }
                )
            .flatMap(//Separate everything into addends for each attribute, and rearrange so that the attribute index is the key 
                {
                  /*case(x, y, s) => bnData.value(x).features.toArray.zip(bnData.value(y).features.toArray).zipWithIndex.map(
                                                                             {case ((x,y),i) if (bnTypes.value(i)) => (i, math.abs(x-y)*s)//Numeric
                                                                             case ((fx,fy),i) => (i, if (fx!=fy) 1.0 else 0.0)})//Nominal
                  */
                  case(x, y, s) => val feat1=lookup.lookup(x).features.toArray
                                  val feat2=lookup.lookup(y).features.toArray
                                  var i = 0;
                                  var res:Array[(Int, Double)] = new Array[(Int, Double)](feat1.length)
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                    {
                                       val range=normalizingDict.value(a)
                                       if (range==0)
                                         res(a)=(a,0.0)
                                       else
                                         res(a)=(a,math.abs(feat1(a)-feat2(a))*s/range)
                                    }
                                    else
                                      if (feat1(a)!=feat2(a))
                                       res(a)=(a,s)//(a,1.0)
                                      else
                                       res(a)=(a,0.0)
                                  res
                }
                )
            .reduceByKey({_ + _})//Sum up for each attribute
            /*.join(rangeAttributes)//In order to divide by the range of each attribute
            .map(//Add 1 to the attribNum so that is in the [1,N] range and divide each result by m and k.
                {
                  case(attribNum, (sum, (min, range))) =>
                    if (bnTypes.value(attribNum))
                     (attribNum+1, sum/(range*numElems))
                    else
                     (attribNum+1, sum/numElems)
                })*/
            .map(//Add 1 to the attribNum so that is in the [1,N] range and divide each result by m
                {
                  case(attribNum, sum) =>
                    (attribNum+1, sum/numElems)
                })
      return dCD;
    }
    
    def selectNumeric(sc: SparkContext, data: RDD[(LabeledPoint, Long)], bnTypes: Broadcast[Array[Boolean]], numNeighbors: Int, normalizingDict: Broadcast[scala.collection.Map[Int, Double]], countsClass: Map[Double, Double], numElems: Double, rangeClass: Double, lshConf:Option[KNiNeConfiguration]): RDD[(Int, Double)] =
    {
      val (kNNGraph,lookup)=if (lshConf.isDefined)
                              getKNNGraphFromKNiNe(sc, data.map(_.swap), numNeighbors, bnTypes, normalizingDict, lshConf.get)
                            else
                              getKNNGraph(sc, data, numNeighbors, bnTypes, normalizingDict)
      /*println(kNNGraph.map(
                {
                  case (index,neighbors) =>
                    (index,neighbors.sortBy(_._2))
                }
                ).sortBy(_._1).first())*/
      val dCD=kNNGraph
            .map(
                {
                  case (index,neighbors) =>
                    (index,neighbors.sortBy(_._2).map({case (idx,distance) => (idx,neighbors.length.toDouble)}))
                }
                )
            /*.map(
                {
                  case(x,distances) =>
                              (x,distances.map({case(y,d) => (y,distances.length)})) //Add the number of neighbors so that we can divide later
                })*/
            .flatMap(//Ungroup everything in order to get closer to addends
                {
                  case (x, distances) =>
                      distances.flatMap({y => List((x,y))})
                })
      dCD.cache()          
      val m_ndc=dCD.map(
                {
                  case (x, (y,k)) => (math.abs(lookup.lookup(x.toInt).label-lookup.lookup(y.toInt).label))
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
                  case(x, (y, s)) => val feat1=lookup.lookup(x).features.toArray
                                  val feat2=lookup.lookup(y).features.toArray
                                  var i = 0;
                                  var res:Array[(Int, (Double,Double))] = new Array[(Int, (Double,Double))](feat1.length)
                                  for( a <- 0 to feat1.length-1)
                                    if (bnTypes.value(a))
                                    {
                                       val range=normalizingDict.value(a)
                                       res(a)=(a,(math.abs(feat1(a)-feat2(a))/range, math.abs(feat1(a)-feat2(a))*math.abs(lookup.lookup(x).label-lookup.lookup(y).label)/range)) //TODO - Class normalization
                                    }
                                    else
                                      if (feat1(a)!=feat2(a))
                                       res(a)=(a,(s,math.abs(lookup.lookup(x).label-lookup.lookup(y).label))) //TODO - Class normalization
                                      else
                                       res(a)=(a,(0.0,0.0))
                                  res
                })
            .reduceByKey({case ((m_nda1, m_ndcda1), (m_nda2, m_ndcda2)) => (m_nda1 + m_nda2, m_ndcda1 + m_ndcda2)})//Sum up for each attribute
            /*.join(rangeAttributes)//In order to divide by the range of each attribute
            .map(//Add 1 to the attribNum so that is in the [1,N] range and compute weights.
                {
                  case(attribNum, ((m_nda, m_ndcda),(min,range))) => (attribNum+1, (m_ndcda/m_ndc - ((rangeClass*m_nda - m_ndcda)/(numNeighborsObtained*rangeClass*numElems-m_ndc)))/range)
                })*/
            .map(//Add 1 to the attribNum so that is in the [1,N] range and compute weights.
                {
                  case(attribNum, (m_nda, m_ndcda)) => (attribNum+1, (m_ndcda/m_ndc - ((rangeClass*m_nda - m_ndcda)/(numNeighborsObtained*rangeClass*numElems-m_ndc))))
                })
      dCD.unpersist(false)
      
      return weights
    }
    
    def main(args: Array[String])
    {
      val options=parseParams(args)
      
      var file=options("dataset").asInstanceOf[String]
      
      var fileOut=options("output").asInstanceOf[String]
      
      //Set up Spark Context
      val conf = new SparkConf()//.setAppName("PruebaReliefF").setMaster("local[8]") //DEBUG!!!!!!!!!!!!!!!!!!!!!!!
      //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      conf.set("spark.eventLog.enabled", "true")
//      conf.set("spark.eventLog.dir","file:///home/eirasf/Escritorio/Tmp-work/sparklog-local")
      
      val sc=new SparkContext(conf)
      //sc.setLogLevel("WARN")//DEBUG!!!!!!!!!!!!!!!!!!!!!!!
      
      val numPartitions:Option[Int]=if (options.contains("num_neighbors"))
                                      Some(options("num_neighbors").asInstanceOf[Double].toInt)
                                    else
                                      None
      
      //Load data from file
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/LargeDatasets/libsvm/isoletTrain.libsvm")
      val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file, -1, numPartitions.getOrElse(3*sc.defaultParallelism))
      //val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "/home/eirasf/Escritorio/Paralelización/Data sets/libsvm/car-mini.libsvm")
      
      //Set maximum number of near neighbors to be taken into account for each instance
      val numNeighbors=options("num_neighbors").asInstanceOf[Double].toInt

      //Set the type (numeric/discrete) for each attribute and class
      val attributeTypes=if (options.contains("attribute_types"))
                          options("attribute_types").asInstanceOf[String].toCharArray().map({case c => c=='N'})
                         else
                          (0 to data.first().features.size-1).map({case c => true}).toArray
      if (attributeTypes.length!=data.first().features.size)
      {
        println("The number of features ("+attributeTypes.length+") does not match the instances in the file ("+data.first().features.size+")")
        return
      }
      
      var discreteClass=(options.contains("class_type")==false) || (options("class_type").asInstanceOf[String].toLowerCase()!="n")
      
      val method=options("method").asInstanceOf[String]
      
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
      println("Method: "+method)
      pw.println("Method: "+method)
      
      val startTime=System.currentTimeMillis()
      
      //Select features
      val kNiNeConf=if (method=="lsh")
                      Some(KNiNeConfiguration.getConfigurationFromOptions(options))
                    else
                      None
      if (kNiNeConf.isDefined)
      {
        println("LSH configuration: "+kNiNeConf.get.toString())
        pw.println("LSH configuration: "+kNiNeConf.get.toString())
      }
      val features=rankFeatures(sc, data, numNeighbors, attributeTypes, discreteClass, kNiNeConf)
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
    
    def showUsageAndExit()=
    {
      println("""Usage: ReliefFFeatureSelector dataset [options]
        Dataset must be a libsvm or text file
    Options:
        -t    Attribute types. String consisting of N or C for each attribute
        -ct    Class type. Either N (numerical) or C (categorical)  
        -k    Number of neighbors (default: """+ReliefFFeatureSelector.DEFAULT_K+""")
        -m    Method used to compute the graph. Valid values: lsh, brute (default: """+ReliefFFeatureSelector.DEFAULT_METHOD+""")
        -r    Starting radius (default: """+LSHKNNGraphBuilder.DEFAULT_RADIUS_START+""")
        -c    Maximum comparisons per item (default: auto)
        -p    Number of partitions for the data RDDs (default: 3*sc.defaultParallelism)
    
    Advanced LSH options:
        -n    Number of hashes per item (default: auto)
        -l    Hash length (default: auto)""")
      System.exit(-1)
    }
    
    def parseParams(p:Array[String]):Map[String, Any]=
    {
      val m=scala.collection.mutable.Map[String, Any]("num_neighbors" -> ReliefFFeatureSelector.DEFAULT_K.toDouble,
                                                      "method" -> ReliefFFeatureSelector.DEFAULT_METHOD,
                                                      "radius_start" -> LSHKNNGraphBuilder.DEFAULT_RADIUS_START)
      if (p.length<=0)
        showUsageAndExit()
      
      m("dataset")=p(0)
      m("output")=p(0).substring(0,p(0).lastIndexOf("."))+"-out.txt"
      
      var i=1
      while (i < p.length)
      {
        if ((i>=p.length-1) || (p(i).charAt(0)!='-'))
        {
          println("Unknown option: "+p(i))
          showUsageAndExit()
        }
        val readOptionName=p(i).substring(1)
        val option=readOptionName match
          {
            case "k"   => "num_neighbors"
            case "m"   => "method"
            case "t"   => "attribute_types"
            case "ct"   => "class_type"
            case "r"   => "radius_start"
            case "n"   => "num_tables"
            case "l"   => "key_length"
            case "c"   => "max_comparisons"
            case "o"   => "output"
            case "p"   => "num_partitions"
            case somethingElse => readOptionName
          }
        if (!m.keySet.exists(_==option) && option==readOptionName)
        {
          println("Unknown option:"+readOptionName)
          showUsageAndExit()
        }
        if (option=="method")
        {
          if (p(i+1)=="lsh" || p(i+1)=="brute")
            m(option)=p(i+1)
          else
          {
            println("Unknown method:"+p(i+1))
            showUsageAndExit()
          }
        }
        else
        {
          if ((option=="class_type") || (option=="attribute_types") || (option=="output"))
            m(option)=p(i+1)
          else
            m(option)=p(i+1).toDouble
        }
        
        i=i+2
      }
      return m.toMap
    }
  }