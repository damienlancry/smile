/*******************************************************************************
 * Copyright (c) 2010-2020 Haifeng Li. All rights reserved.
 *
 * Smile is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * Smile is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Smile.  If not, see <https://www.gnu.org/licenses/>.
 ******************************************************************************/

package smile.classification;

import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.vector.BaseVector;
import smile.math.MathEx;
import smile.util.IntSet;

import java.util.function.BiFunction;

/**
 * Positive Unlabelled learner for reducing the problem of
 * multi positive unlabelled learning to multiple binary classification problems.
 * It involves training a single classifier per positive class, with the samples
 * of that class as positive samples and all other samples
 * (examples of other positive classes, and unlabelled points) as negatives.
 * This strategy requires the base classifiers to produce a real-valued
 * confidence score for its decision, rather than just a class label;
 * discrete class labels alone can lead to ambiguities, where multiple
 * classes are predicted for a single sample.
 * <p>
 * At prediction time, all classifiers are applied to an unseen sample
 * x and the predicted label k is the one corresponding to the highest
 * confidence score if it is greater than a threshold e.g. 0.5 or the
 * negative class otherwise.
 *
 * @author Damien Lancry
 */
public class PositiveUnlabelled<T> implements SoftClassifier<Tuple>, DataFrameClassifier {
    private static final long serialVersionUID = 2L;

    /** The number of positive classes. */
    private int k;
    /** The binary classifiers of each positive class. */
    private Classifier<T>[] classifiers;
    /** The class label encoder. */
    private IntSet labels;

    /**
     * Constructor.
     * @param classifiers the binary classifier for each class.
     */
    public PositiveUnlabelled(Classifier<T>[] classifiers) {
        this(classifiers, IntSet.of(classifiers.length + 1));
    }

    /**
     * Constructor.
     * @param classifiers the binary classifier for each positive class.
     * @param labels the class labels.
     */
    public PositiveUnlabelled(Classifier<T>[] classifiers, IntSet labels) {
        this.classifiers = classifiers;
        this. k = classifiers.length;
        this.labels = labels;
    }

    /**
     * Fits one binary model per positive class.
     * Use +1 and -1 as positive and negative class labels.
     * @param formula a symbolic description of the model to be fitted.
     * @param data the data frame of the explanatory and response variables. the response variable is assumed to be -1 for all unlabelled point.
     * @param trainer the lambda to train binary classifiers.
     */
    public static PositiveUnlabelled<Tuple> fit(Formula formula, DataFrame data, BiFunction<Formula, DataFrame, SoftClassifier<Tuple>> trainer) {
        return fit(formula, data, +1, -1, trainer);
    }

    /**
     * Fits one binary model per positive class.
     * @param formula a symbolic description of the model to be fitted.
     * @param data the data frame of the explanatory and response variables. the response variable is assumed to be -1 for all unlabelled point.
     * @param pos the class label for one case.
     * @param neg the class label for rest cases.
     * @param trainer the lambda to train binary classifiers.
     */
    public static PositiveUnlabelled<Tuple> fit(Formula formula, DataFrame data, int pos, int neg, BiFunction<Formula, DataFrame, SoftClassifier<Tuple>> trainer) {
        formula = formula.expand(data.schema());
        DataFrame x = formula.x(data);
        BaseVector bv = formula.y(data);

        ClassLabels codec = ClassLabels.fit(bv);
        if (codec.labels.min != -1) {
            throw new IllegalArgumentException("There is no unlabelled data in the training set");
        }
        int k = codec.k - 1;
        if (k == 0) {
            throw new IllegalArgumentException(String.format("Only %d positive classes", k));
        }
        int n = x.nrows();
        int[] y = codec.y;

        SoftClassifier[] classifiers = new SoftClassifier[k];
        for (int i = 0; i < k; i++) {
            int[][] yi = new int[n][1];
            for (int j = 0; j < n; j++) {
                yi[j][0] = y[j] == i + 1 ? pos : neg;
            }

            classifiers[i] = trainer.apply(formula, x.merge(DataFrame.of(yi, bv.name())));
        }
        return new PositiveUnlabelled<Tuple>(classifiers, codec.labels);
    }

    @Override
    public int predict(Tuple x) {
        int y = 0;
        double maxp = 0.0;
        for (int i = 0; i < k; i++) {
            double[] proba = new double[2];
            ((SoftClassifier<Tuple>) classifiers[i]).predict(x, proba);
            double p = proba[1];
            if (p > maxp && p > 0.5) {
                y = i + 1;
                maxp = p;
            }
        }
        return labels.valueOf(y);
    }

    public int k() {
        return this.k;
    }

    public int[] labels() {
        return this.labels.values;
    }

    @Override
    public Formula formula() {
        return ((DataFrameClassifier) classifiers[0]).formula();
    }

    @Override
    public StructType schema() {
        return ((DataFrameClassifier) classifiers[0]).schema();
    }

    @Override
    public int predict(Tuple x, double[] posteriori) {
        int y = 0;
        double maxp = 0.0;
        for (int i = 0; i < k; i++) {
            double[] proba = new double[2];
            ((SoftClassifier<Tuple>) classifiers[i]).predict(x, proba);
            double p = proba[1];
            posteriori[i] = p;
            if (p > maxp && p > 0.5) {
                y = i + 1;
                maxp = p;
            }
        }
        return labels.valueOf(y);
    }
}
