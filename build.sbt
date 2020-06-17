name := "spark-infotheoretic-feature-selection"

version := "0.1"

organization := "com.github.sramirez"

scalaVersion := "2.11.11"

unmanagedJars in Compile += file("lib/knine-assembly-0.1.jar")
//unmanagedJars in Compile += file("lib/spark-knn-0.2.0.jar")

resolvers ++= Seq(
  "Apache Staging" at "https://repository.apache.org/content/repositories/staging/",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"
)

publishMavenStyle := true

sparkPackageName := "sramirez/infotheoretic-feature-selection"

sparkVersion := "2.4.0"

sparkComponents += "mllib"
