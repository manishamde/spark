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

package org.apache.spark.mllib.tree.configuration

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.FeatureSubsetStrategy._


/**
 * :: Experimental ::
 * Stores all the configuration options for tree construction
 * @param algo classification or regression
 * @param impurity criterion used for information gain calculation
 * @param maxDepth maximum depth of the tree
 * @param numClassesForClassification number of classes for classification. Default value is 2
 *                                    leads to binary classification
 * @param maxBins maximum number of bins used for splitting features
 * @param quantileCalculationStrategy algorithm for calculating quantiles
 * @param categoricalFeaturesInfo A map storing information about the categorical variables and the
 *                                number of discrete values they take. For example, an entry (n ->
 *                                k) implies the feature n is categorical with k categories 0,
 *                                1, 2, ... , k-1. It's important to note that features are
 *                                zero-indexed.
 * @param maxMemoryInMB maximum memory in MB allocated to histogram aggregation. Default value is
 *                      128 MB.
 * @param labelWeights A map storing weights applied to each label for handling unbalanced
 *                     datasets. For example, an entry (n -> k) implies the a weight of k is
 *                     applied to an instance with label n. It's important to note that labels
 *                     are zero-index and take values 0, 1, 2, ... , numClasses.
 * @param boostingIterations Number of iterations for boosting algorithms
 * @param featureSubsetStrategy Indicates the features used for training a tree
 * @param isRandomForest Indicates whether a random forest is being created
 *
 *
 */
@Experimental
class Strategy (
    val algo: Algo,
    val impurity: Impurity,
    val maxDepth: Int,
    val numClassesForClassification: Int = 2,
    val maxBins: Int = 100,
    val quantileCalculationStrategy: QuantileStrategy = Sort,
    val categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int](),
    val maxMemoryInMB: Int = 128,
    val labelWeights: Map[Int, Double] = Map[Int, Double](),
    val boostingIterations: Int = 100,
    val numTrees: Int = 1,
    val featureSubsetStrategy: FeatureSubsetStrategy = All,
    val isRandomForest: Boolean = false) extends Serializable {

  require(numClassesForClassification >= 2)
  if (isRandomForest) {
    require(numTrees > 1)
    require(labelWeights.size == 0)
  }
  val isMulticlassClassification = numClassesForClassification > 2
  val isMulticlassWithCategoricalFeatures
    = isMulticlassClassification && (categoricalFeaturesInfo.size > 0)

}
