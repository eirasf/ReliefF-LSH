name := "spark-infotheoretic-feature-selection"

version := "0.1"

organization := "com.github.sramirez"

scalaVersion := "2.10.4"

unmanagedJars in Compile += file("lib/spark-knine-0.2.jar")

resolvers ++= Seq(
  "Apache Staging" at "https://repository.apache.org/content/repositories/staging/",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"
)

publishMavenStyle := true

sparkPackageName := "sramirez/infotheoretic-feature-selection"

sparkVersion := "1.4.0"

sparkComponents += "mllib"
