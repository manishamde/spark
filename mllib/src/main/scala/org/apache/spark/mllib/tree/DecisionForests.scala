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

import scala.collection.JavaConverters._

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.{Impurities, Impurity}
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.api.java.JavaRDD

object DecisionForests {


  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param strategy The configuration parameters for the tree algorithm which specify the type
   *                 of algorithm (classification, regression, etc.), feature type (continuous,
   *                 categorical), depth of the tree, quantile calculation strategy, etc.
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(input: RDD[LabeledPoint], strategy: Strategy): DecisionTreeModel = {
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo algorithm, classification or regression
   * @param impurity impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @param numTrees number of trees to train in parallel for random forests
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
             input: RDD[LabeledPoint],
             algo: Algo,
             impurity: Impurity,
             maxDepth: Int,
             numTrees: Int): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo algorithm, classification or regression
   * @param impurity impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @param numClassesForClassification number of classes for classification. Default value of 2.
   * @param numTrees number of trees to train in parallel for random forests
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
             input: RDD[LabeledPoint],
             algo: Algo,
             impurity: Impurity,
             maxDepth: Int,
             numClassesForClassification: Int,
             numTrees: Int): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth, numClassesForClassification)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model.
   * The method supports binary and multiclass classification and regression.
   *
   * Note: Using [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   *       and [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   *       is recommended to clearly separate classification and regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param algo classification or regression
   * @param impurity criterion used for information gain calculation
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   * @param numClassesForClassification number of classes for classification. Default value of 2.
   * @param maxBins maximum number of bins used for splitting features
   * @param quantileCalculationStrategy  algorithm for calculating quantiles
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @param numTrees number of trees to train in parallel for random forests
   * @return DecisionTreeModel that can be used for prediction
   */
  def train(
             input: RDD[LabeledPoint],
             algo: Algo,
             impurity: Impurity,
             maxDepth: Int,
             numClassesForClassification: Int,
             maxBins: Int,
             quantileCalculationStrategy: QuantileStrategy,
             categoricalFeaturesInfo: Map[Int,Int],
             numTrees: Int): DecisionTreeModel = {
    val strategy = new Strategy(algo, impurity, maxDepth, numClassesForClassification, maxBins,
      quantileCalculationStrategy, categoricalFeaturesInfo)
    new DecisionTree(strategy).train(input)
  }

  /**
   * Method to train a decision tree model for binary or multiclass classification.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              Labels should take values {0, 1, ..., numClasses-1}.
   * @param numClassesForClassification number of classes for classification.
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @param impurity Criterion used for information gain calculation.
   *                 Supported values: "gini" (recommended) or "entropy".
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   *                  (suggested value: 4)
   * @param maxBins maximum number of bins used for splitting features
   *                 (suggested value: 100)
   * @param numTrees number of trees to train in parallel for random forests
   * @return DecisionTreeModel that can be used for prediction
   */
  def trainClassifier(
                       input: RDD[LabeledPoint],
                       numClassesForClassification: Int,
                       categoricalFeaturesInfo: Map[Int, Int],
                       impurity: String,
                       maxDepth: Int,
                       maxBins: Int,
                       numTrees: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity)
    train(input, Classification, impurityType, maxDepth, numClassesForClassification, maxBins, Sort,
      categoricalFeaturesInfo, numTrees)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.DecisionTree$#trainClassifier]]
   */
  def trainClassifier(
                       input: JavaRDD[LabeledPoint],
                       numClassesForClassification: Int,
                       categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer],
                       impurity: String,
                       maxDepth: Int,
                       maxBins: Int,
                       numTrees: Int): DecisionTreeModel = {
    trainClassifier(input.rdd, numClassesForClassification,
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      impurity, maxDepth, maxBins, numTrees)
  }

  /**
   * Method to train a decision tree model for regression.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.LabeledPoint]].
   *              Labels are real numbers.
   * @param categoricalFeaturesInfo Map storing arity of categorical features.
   *                                E.g., an entry (n -> k) indicates that feature n is categorical
   *                                with k categories indexed from 0: {0, 1, ..., k-1}.
   * @param impurity Criterion used for information gain calculation.
   *                 Supported values: "variance".
   * @param maxDepth Maximum depth of the tree.
   *                 E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
   *                  (suggested value: 4)
   * @param maxBins maximum number of bins used for splitting features
   *                 (suggested value: 100)
   * @param numTrees number of trees to train in parallel for random forests
   * @return DecisionTreeModel that can be used for prediction
   */
  def trainRegressor(
                      input: RDD[LabeledPoint],
                      categoricalFeaturesInfo: Map[Int, Int],
                      impurity: String,
                      maxDepth: Int,
                      maxBins: Int,
                      numTrees: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity)
    train(input, Regression, impurityType, maxDepth, 0, maxBins, Sort, categoricalFeaturesInfo, numTrees)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.DecisionTree$#trainRegressor]]
   */
  def trainRegressor(
                      input: JavaRDD[LabeledPoint],
                      categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer],
                      impurity: String,
                      maxDepth: Int,
                      maxBins: Int,
                      numTrees: Int): DecisionTreeModel = {
    trainRegressor(input.rdd,
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap,
      impurity, maxDepth, maxBins, numTrees)
  }


}
