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

package org.apache.spark.mllib.tree.model

import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.model.Model

/**
 * :: Experimental ::
 * Represents a multiclass classification model for AdaBoost algorithm that predicts to which of a
 * set of categories an example belongs. The categories are represented by double values: 0.0,
 * 1.0, 2.0, etc.
 */
@Experimental
class MultiClassAdaModel(
    alphas: Array[Double],
    trees: Array[Model],
    strategy: Strategy) extends Model {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return predicted category from the trained model
   */
  override def predict(testData: Vector): Double = {
    val K = strategy.numClassesForClassification
    val M = strategy.boostingIterations
    val treePredictionsPerClassAndIteraton = for {
      k <- 0 until K
      m <- 0 until M
      alpha = alphas(m)
      treePredict = trees(m).predict(testData)
    } yield ((k, alpha * equalIdentity(k,treePredict)))

    // Find prediction class
    treePredictionsPerClassAndIteraton
    .groupBy(_._1)
    .mapValues(x => x.map(_._2).sum)
    .reduce((x,y) => if (x._2 < y._2) y else x)
    ._1
  }

  private def equalIdentity(a: Double, b: Double): Double = {
    if (a == b) 1.0 else 0.0
  }
}
