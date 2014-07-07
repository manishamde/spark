/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext._

import org.apache.spark.mllib.util.{MLUtils, LinearDataGenerator}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.model.Model
import org.apache.spark.mllib.point.WeightedLabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.point.WeightedLabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint


object GBTRunner {

  def main(args: Array[String]) {
    val sc = new SparkContext("local", "DecisionTree")
    //val fileName = "/Users/manish/Documents/code/mere_projects/data/cov_type/covtype.csv"
    //val dataset = "covtype"
    val fileName = "/Users/manish/Documents/code/mere_projects/data/breast_cancer/bc.csv"
    val dataset = "breast_cancer"
    val input = loadLabeledData(sc, fileName, dataset)
//    val input = LinearDataGenerator.generateLinearRDD(sc,10000,20,10)
//    val input = MLUtils.loadLabeledData(sc,
//      "/Users/manish/Documents/code/mere_projects/spark/mllib/data/ridge-data/lpsa.data")

    val weightedInput = input.map(x => new WeightedLabeledPoint(x.label,x.features))
    val strategy = new Strategy(algo = Regression, impurity = Variance,
      maxDepth = 4, maxBins = 100, boostingIterations = 100)
    val tic = System.currentTimeMillis()
    val model = new GradientBoost(strategy).train(weightedInput)
    val toc = System.currentTimeMillis()
    val trainingTime = (toc - tic)/1000
    println(s"training time = $trainingTime")
    val mse = meanSquaredError(model, input)
    println("mean square error for GBT = " + mse)

    val ticLin = System.currentTimeMillis()
    val linearModel = LassoWithSGD.train(input, 100)
    val tocLin = System.currentTimeMillis()
    val trainingTimeRidge = (tocLin - ticLin) / 1000
    println(s"training time for ridge regression = $trainingTimeRidge")
    val linearMse = meanSquaredError(linearModel, input)
    println("mean square error for ridge regression = " + linearMse)

    val ticTree = System.currentTimeMillis()
    val treeModel = DecisionTree.train(input, strategy)
    val tocTree = System.currentTimeMillis()
    val trainingTimeTree = (tocTree - ticTree) / 1000
    println(s"training time for decision tree regression = $trainingTimeTree")
    val treeMse = meanSquaredError(treeModel, input)
    println("mean square error for decision tree regression = " + treeMse)


  }

  // TODO: Port this method to a generic metrics package
  /**
   * Calculates the mean squared error for regression.
   */
  private def meanSquaredError(tree: Model, data: RDD[LabeledPoint]): Double = {
    data.map { y =>
      val err = tree.predict(y.features) - y.label
      err * err
    }.mean()
  }

  // TODO: Port this method to a generic metrics package
  /**
   * Calculates the mean squared error for regression.
   */
  private def meanSquaredError(tree: RegressionModel, data: RDD[LabeledPoint]): Double = {
    data.map { y =>
      val err = tree.predict(y.features) - y.label
      err * err
    }.mean()
  }


  /**
   * Load labeled data from a file. The data format used here is
   * <L>, <f1> <f2> ...,
   * where <f1>, <f2> are feature values in Double and <L> is the corresponding label as Double.
   *
   * @param sc SparkContext
   * @param dir Directory to the input data files.
   * @return An RDD of LabeledPoint. Each labeled point has two elements: the first element is
   *         the label, and the second element represents the feature values (an array of Double).
   */
  def loadLabeledData(sc: SparkContext, dir: String, dataset: String): RDD[LabeledPoint] = {
    sc.textFile(dir).map { line =>
      val parts = line.trim().split(",")
      if (dataset == "random") {
        val label = parts(0).toDouble
        val features = parts.slice(0, parts.length - 1).map(_.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      } else {
        val label = parts(0).toDouble
        val features = parts.slice(1, parts.length).map(_.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }
    }
  }



}
