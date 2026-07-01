/**
 * Browser-based Machine Learning Engine in TypeScript.
 * Includes data scaling, feature encoding, Multiple Linear Regression, CART Decision Trees,
 * and Random Forest (Regressor & Classifier), plus evaluation metrics.
 */

// --- 1. PREPROCESSING & ENCODING ---

export class MinMaxScaler {
  private mins: Record<string, number> = {};
  private maxs: Record<string, number> = {};

  fit(data: Record<string, any>[], features: string[]) {
    features.forEach((feat) => {
      const vals = data.map((d) => Number(d[feat])).filter((v) => !isNaN(v));
      if (vals.length > 0) {
        this.mins[feat] = Math.min(...vals);
        this.maxs[feat] = Math.max(...vals);
      } else {
        this.mins[feat] = 0;
        this.maxs[feat] = 1;
      }
    });
  }

  transform(row: Record<string, any>, features: string[]): Record<string, number> {
    const scaled: Record<string, number> = {};
    features.forEach((feat) => {
      const val = Number(row[feat]);
      const min = this.mins[feat] ?? 0;
      const max = this.maxs[feat] ?? 1;
      const denom = max - min;
      scaled[feat] = denom === 0 ? 0 : (val - min) / denom;
    });
    return scaled;
  }
}

export class StandardScaler {
  private means: Record<string, number> = {};
  private stds: Record<string, number> = {};

  fit(data: Record<string, any>[], features: string[]) {
    features.forEach((feat) => {
      const vals = data.map((d) => Number(d[feat])).filter((v) => !isNaN(v));
      if (vals.length > 0) {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const variance = vals.map((v) => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / vals.length;
        this.means[feat] = mean;
        this.stds[feat] = Math.sqrt(variance) || 1;
      } else {
        this.means[feat] = 0;
        this.stds[feat] = 1;
      }
    });
  }

  transform(row: Record<string, any>, features: string[]): Record<string, number> {
    const scaled: Record<string, number> = {};
    features.forEach((feat) => {
      const val = Number(row[feat]);
      const mean = this.means[feat] ?? 0;
      const std = this.stds[feat] ?? 1;
      scaled[feat] = (val - mean) / std;
    });
    return scaled;
  }
}

export class LabelEncoder {
  private mapping: Record<string, Record<string, number>> = {};
  private reverseMapping: Record<string, Record<number, string>> = {};

  fit(data: Record<string, any>[], features: string[]) {
    features.forEach((feat) => {
      const vals = Array.from(new Set(data.map((d) => String(d[feat]))));
      this.mapping[feat] = {};
      this.reverseMapping[feat] = {};
      vals.forEach((val, idx) => {
        this.mapping[feat][val] = idx;
        this.reverseMapping[feat][idx] = val;
      });
    });
  }

  transform(row: Record<string, any>, features: string[]): Record<string, number> {
    const encoded: Record<string, number> = {};
    features.forEach((feat) => {
      const val = String(row[feat]);
      encoded[feat] = this.mapping[feat]?.[val] ?? 0;
    });
    return encoded;
  }

  inverseTransform(feat: string, val: number): string {
    return this.reverseMapping[feat]?.[val] ?? String(val);
  }
}

// --- 2. TRAIN-TEST SPLIT ---

export function trainTestSplit(
  data: Record<string, any>[],
  testRatio = 0.2,
  shuffle = true
): { train: Record<string, any>[]; test: Record<string, any>[] } {
  let indices = Array.from({ length: data.length }, (_, i) => i);
  if (shuffle) {
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }
  const splitIndex = Math.floor(data.length * (1 - testRatio));
  const trainIndices = indices.slice(0, splitIndex);
  const testIndices = indices.slice(splitIndex);

  return {
    train: trainIndices.map((i) => data[i]),
    test: testIndices.map((i) => data[i]),
  };
}

// --- 3. MULTIPLE LINEAR REGRESSION ---

export class MultipleLinearRegression {
  public weights: number[] = [];
  public bias = 0;

  fit(X: number[][], y: number[], epochs = 1000, lr = 0.01, onProgress?: (epoch: number, loss: number) => void) {
    const numSamples = X.length;
    if (numSamples === 0) return;
    const numFeatures = X[0].length;

    this.weights = new Array(numFeatures).fill(0).map(() => Math.random() - 0.5);
    this.bias = Math.random() - 0.5;

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let dW = new Array(numFeatures).fill(0);
      let dB = 0;
      let loss = 0;

      for (let i = 0; i < numSamples; i++) {
        let pred = this.bias;
        for (let j = 0; j < numFeatures; j++) {
          pred += X[i][j] * this.weights[j];
        }

        const err = pred - y[i];
        loss += Math.pow(err, 2);

        for (let j = 0; j < numFeatures; j++) {
          dW[j] += err * X[i][j];
        }
        dB += err;
      }

      loss = loss / (2 * numSamples);

      // Gradient updates
      for (let j = 0; j < numFeatures; j++) {
        this.weights[j] -= (lr * dW[j]) / numSamples;
      }
      this.bias -= (lr * dB) / numSamples;

      if (epoch % 50 === 0 || epoch === epochs) {
        if (onProgress) onProgress(epoch, loss);
      }
    }
  }

  predict(X_row: number[]): number {
    return this.bias + X_row.reduce((sum, val, idx) => sum + val * (this.weights[idx] ?? 0), 0);
  }
}

// --- 4. DECISION TREE (CART) ---

interface DecisionNode {
  feature?: string;
  threshold?: number;
  left?: DecisionNode;
  right?: DecisionNode;
  value?: number; // Represent prediction for leaves
  isLeaf: boolean;
}

export class DecisionTree {
  private root: DecisionNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private isClassifier: boolean;

  constructor(isClassifier = false, maxDepth = 5, minSamplesSplit = 2) {
    this.isClassifier = isClassifier;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(data: Record<string, any>[], features: string[], target: string) {
    this.root = this.buildTree(data, features, target, 0);
  }

  predict(row: Record<string, any>): number {
    if (!this.root) return 0;
    return this.traverseTree(this.root, row);
  }

  private buildTree(data: Record<string, any>[], features: string[], target: string, depth: number): DecisionNode {
    const numSamples = data.length;
    const values = data.map((d) => d[target]);

    const isAllSame = new Set(values).size === 1;
    if (numSamples < this.minSamplesSplit || depth >= this.maxDepth || isAllSame) {
      return { isLeaf: true, value: this.calculateLeafValue(values) };
    }

    let bestGain = -1;
    let bestFeature: string | undefined;
    let bestThreshold: number | undefined;
    let bestLeft: Record<string, any>[] = [];
    let bestRight: Record<string, any>[] = [];

    features.forEach((feat) => {
      const thresholds = Array.from(new Set(data.map((d) => d[feat])));
      thresholds.forEach((thresh) => {
        const left = data.filter((d) => d[feat] <= thresh);
        const right = data.filter((d) => d[feat] > thresh);

        if (left.length === 0 || right.length === 0) return;

        const gain = this.calculateInformationGain(values, left.map((d) => d[target]), right.map((d) => d[target]));
        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feat;
          bestThreshold = thresh;
          bestLeft = left;
          bestRight = right;
        }
      });
    });

    if (bestGain <= 0 || !bestFeature || bestThreshold === undefined) {
      return { isLeaf: true, value: this.calculateLeafValue(values) };
    }

    const leftChild = this.buildTree(bestLeft, features, target, depth + 1);
    const rightChild = this.buildTree(bestRight, features, target, depth + 1);

    return {
      isLeaf: false,
      feature: bestFeature,
      threshold: bestThreshold,
      left: leftChild,
      right: rightChild,
    };
  }

  private calculateLeafValue(values: any[]): number {
    if (this.isClassifier) {
      // Return mode for classification
      const counts: Record<string, number> = {};
      values.forEach((v) => {
        counts[v] = (counts[v] ?? 0) + 1;
      });
      let maxCount = -1;
      let mode = 0;
      Object.entries(counts).forEach(([val, count]) => {
        if (count > maxCount) {
          maxCount = count;
          mode = Number(val);
        }
      });
      return mode;
    } else {
      // Return mean for regression
      return values.reduce((a, b) => a + b, 0) / values.length;
    }
  }

  private calculateInformationGain(parent: any[], left: any[], right: any[]): number {
    if (this.isClassifier) {
      // Gini Gain
      const giniParent = this.calculateGini(parent);
      const giniLeft = this.calculateGini(left);
      const giniRight = this.calculateGini(right);
      const pLeft = left.length / parent.length;
      const pRight = right.length / parent.length;
      return giniParent - (pLeft * giniLeft + pRight * giniRight);
    } else {
      // Variance Reduction Gain
      const varParent = this.calculateVariance(parent);
      const varLeft = this.calculateVariance(left);
      const varRight = this.calculateVariance(right);
      const pLeft = left.length / parent.length;
      const pRight = right.length / parent.length;
      return varParent - (pLeft * varLeft + pRight * varRight);
    }
  }

  private calculateGini(values: any[]): number {
    const length = values.length;
    if (length === 0) return 0;
    const counts: Record<string, number> = {};
    values.forEach((v) => {
      counts[v] = (counts[v] ?? 0) + 1;
    });
    let sumSqProb = 0;
    Object.values(counts).forEach((c) => {
      sumSqProb += Math.pow(c / length, 2);
    });
    return 1 - sumSqProb;
  }

  private calculateVariance(values: number[]): number {
    const length = values.length;
    if (length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / length;
    return values.map((v) => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / length;
  }

  private traverseTree(node: DecisionNode, row: Record<string, any>): number {
    if (node.isLeaf) return node.value!;
    const val = row[node.feature!];
    if (val <= node.threshold!) {
      return this.traverseTree(node.left!, row);
    } else {
      return this.traverseTree(node.right!, row);
    }
  }
}

// --- 5. RANDOM FOREST (Regressor & Classifier) ---

export class RandomForest {
  private trees: DecisionTree[] = [];
  private numEstimators: number;
  private isClassifier: boolean;
  private maxDepth: number;
  private minSamplesSplit: number;
  public featureImportances: Record<string, number> = {};

  constructor(isClassifier = false, numEstimators = 10, maxDepth = 5, minSamplesSplit = 2) {
    this.isClassifier = isClassifier;
    this.numEstimators = numEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(data: Record<string, any>[], features: string[], target: string) {
    this.trees = [];
    this.featureImportances = {};
    features.forEach((f) => (this.featureImportances[f] = 0));

    for (let i = 0; i < this.numEstimators; i++) {
      const tree = new DecisionTree(this.isClassifier, this.maxDepth, this.minSamplesSplit);
      
      // Bagging: Bootstrap sampling
      const bootstrappedData: Record<string, any>[] = [];
      for (let j = 0; j < data.length; j++) {
        const randIdx = Math.floor(Math.random() * data.length);
        bootstrappedData.push(data[randIdx]);
      }

      // Feature bagging: Random subset of features
      const m = Math.max(1, Math.floor(Math.sqrt(features.length)));
      const shuffledFeats = [...features].sort(() => Math.random() - 0.5);
      const featureSubset = shuffledFeats.slice(0, m);

      tree.fit(bootstrappedData, featureSubset, target);
      this.trees.push(tree);

      // Track feature usage for mock feature importance
      featureSubset.forEach((feat) => {
        this.featureImportances[feat] += 1 / this.numEstimators;
      });
    }

    // Normalize feature importances
    const totalImp = Object.values(this.featureImportances).reduce((a, b) => a + b, 0);
    if (totalImp > 0) {
      Object.keys(this.featureImportances).forEach((feat) => {
        this.featureImportances[feat] /= totalImp;
      });
    }
  }

  predict(row: Record<string, any>): number {
    const predictions = this.trees.map((tree) => tree.predict(row));
    if (this.isClassifier) {
      // Majority voting
      const counts: Record<number, number> = {};
      predictions.forEach((p) => {
        counts[p] = (counts[p] ?? 0) + 1;
      });
      let maxCount = -1;
      let winner = 0;
      Object.entries(counts).forEach(([val, count]) => {
        if (count > maxCount) {
          maxCount = count;
          winner = Number(val);
        }
      });
      return winner;
    } else {
      // Average prediction
      return predictions.reduce((a, b) => a + b, 0) / predictions.length;
    }
  }
}

// --- 6. METRICS & ANALYSIS ---

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
}

export function evaluateRegression(actuals: number[], predictions: number[]): RegressionMetrics {
  const n = actuals.length;
  if (n === 0) return { mae: 0, mse: 0, rmse: 0, r2: 0 };

  const mae = actuals.reduce((sum, actual, i) => sum + Math.abs(actual - predictions[i]), 0) / n;
  const mse = actuals.reduce((sum, actual, i) => sum + Math.pow(actual - predictions[i], 2), 0) / n;
  const rmse = Math.sqrt(mse);

  const meanActual = actuals.reduce((a, b) => a + b, 0) / n;
  const ssTotal = actuals.reduce((sum, actual) => sum + Math.pow(actual - meanActual, 2), 0);
  const ssRes = actuals.reduce((sum, actual, i) => sum + Math.pow(actual - predictions[i], 2), 0);

  const r2 = ssTotal === 0 ? 1 : 1 - ssRes / ssTotal;

  return { mae, mse, rmse, r2 };
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: Record<string, Record<string, number>>;
}

export function evaluateClassification(actuals: any[], predictions: any[]): ClassificationMetrics {
  const n = actuals.length;
  if (n === 0) return { accuracy: 0, precision: 0, recall: 0, f1: 0, confusionMatrix: {} };

  const uniqueClasses = Array.from(new Set([...actuals, ...predictions])).map(String);
  const confusionMatrix: Record<string, Record<string, number>> = {};

  uniqueClasses.forEach((c1) => {
    confusionMatrix[c1] = {};
    uniqueClasses.forEach((c2) => {
      confusionMatrix[c1][c2] = 0;
    });
  });

  let correct = 0;
  for (let i = 0; i < n; i++) {
    const act = String(actuals[i]);
    const pred = String(predictions[i]);
    if (confusionMatrix[act] && confusionMatrix[act][pred] !== undefined) {
      confusionMatrix[act][pred]++;
    }
    if (act === pred) correct++;
  }

  const accuracy = correct / n;

  // Multi-class macro-averaged metrics
  let totalPrecision = 0;
  let totalRecall = 0;
  let activeClassesCount = 0;

  uniqueClasses.forEach((cls) => {
    let tp = 0;
    let fp = 0;
    let fn = 0;

    uniqueClasses.forEach((actualCls) => {
      uniqueClasses.forEach((predCls) => {
        const count = confusionMatrix[actualCls][predCls];
        if (cls === actualCls && cls === predCls) tp += count;
        else if (cls === predCls) fp += count;
        else if (cls === actualCls) fn += count;
      });
    });

    const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
    const rec = tp + fn === 0 ? 0 : tp / (tp + fn);

    if (tp + fp > 0 || tp + fn > 0) {
      totalPrecision += prec;
      totalRecall += rec;
      activeClassesCount++;
    }
  });

  const precision = activeClassesCount === 0 ? 0 : totalPrecision / activeClassesCount;
  const recall = activeClassesCount === 0 ? 0 : totalRecall / activeClassesCount;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

  return {
    accuracy,
    precision,
    recall,
    f1,
    confusionMatrix,
  };
}

/**
 * Calculates a Pearson Correlation Matrix for numeric features
 */
export function getCorrelationMatrix(data: Record<string, any>[], features: string[]): Record<string, Record<string, number>> {
  const matrix: Record<string, Record<string, number>> = {};
  const cleanData = data.map((row) => {
    const cleanRow: Record<string, number> = {};
    features.forEach((feat) => {
      cleanRow[feat] = Number(row[feat]);
    });
    return cleanRow;
  });

  features.forEach((f1) => {
    matrix[f1] = {};
    features.forEach((f2) => {
      if (f1 === f2) {
        matrix[f1][f2] = 1.0;
        return;
      }

      const x = cleanData.map((d) => d[f1]).filter((v) => !isNaN(v));
      const y = cleanData.map((d) => d[f2]).filter((v) => !isNaN(v));

      if (x.length === 0 || y.length === 0 || x.length !== y.length) {
        matrix[f1][f2] = 0;
        return;
      }

      const n = x.length;
      const meanX = x.reduce((a, b) => a + b, 0) / n;
      const meanY = y.reduce((a, b) => a + b, 0) / n;

      const num = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
      const denX = Math.sqrt(x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0));
      const denY = Math.sqrt(y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0));

      const corr = denX * denY === 0 ? 0 : num / (denX * denY);
      matrix[f1][f2] = Number(corr.toFixed(4));
    });
  });

  return matrix;
}
