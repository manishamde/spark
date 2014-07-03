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

import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.model.Model

object DecisionTreeRunner extends Logging {


  val usage = """
    Usage: DecisionTreeRunner <master>[slices] --algo <Classification,
    Regression> --trainDataDir path --testDataDir path --maxDepth num [--impurity <Gini,Entropy,
    Variance>] [--maxBins num]
              """

  def main(args: Array[String]) {

    val sc = new SparkContext("local", "DecisionTree")
    // Load test data.
    //val fileName = "/Users/manish/Documents/code/mere_projects/spark/mllib/data/iris.csv"
    val fileName = "/Users/manish/Documents/code/mere_projects/data/cov_type/covtype.csv"
    val dataset = "covtype"
    val trainData = loadLabeledData(sc, fileName, dataset)
    val testData = loadLabeledData(sc, fileName, dataset)
    val algo = Classification
    //val strategy = new Strategy(Classification, Gini, 3, 3)
    val strategy = new Strategy(Classification, Gini, 3, 7, 300)
    val model = DecisionTree.train(trainData, strategy)

    // Measure algorithm accuracy
    if (algo == Classification) {
      val accuracy = accuracyScore(model, testData)
      logDebug("accuracy = " + accuracy)
    }

    if (algo == Regression) {
      val mse = meanSquaredError(model, testData)
      logDebug("mean square error = " + mse)
    }

    sc.stop()
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
      if (dataset == "iris") {
        val label = parts(parts.length - 1).toDouble
        val features = parts.slice(0, parts.length - 1).map(_.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      } else {
        val label = parts(0).toDouble
        val features = parts.slice(1, parts.length).map(_.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }
    }
  }


  /**
   * Calculates the classifier accuracy.
   */
  private def accuracyScore(model: Model, data: RDD[LabeledPoint],
                            threshold: Double = 0.5): Double = {
    val correctCount = data.filter(y => model.predict(y.features) == y.label).count()
    val count = data.count()
    logDebug("correct prediction count = " +  correctCount)
    logDebug("data count = " + count)
    correctCount.toDouble / count
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

}
