/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

/**
 *
 * @author jelica
 */
public class WinesJAN2018 {

    int inputCount = 13;
    int outputCount = 3;
    double[] learningRates = {0.2, 0.4, 0.6};
    ArrayList<Training> trainings = new ArrayList<>();

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new WinesJAN2018().run();
    }

    private void run() {
        System.out.println("Ucitavanje dataseta: ");
        DataSet dataset = DataSet.createFromFile("wines.csv", inputCount, outputCount, ",");
        MaxNormalizer normalizer = new MaxNormalizer(dataset);
        normalizer.normalize(dataset);
        dataset.shuffle();

        DataSet[] trainAndTest = dataset.createTrainingAndTestSubsets(0.70, 0.30);
        DataSet train = trainAndTest[0];
        DataSet test = trainAndTest[1];

        int numberOfIterations = 0;

        for (double lr : learningRates) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, 20, outputCount);
            BackPropagation bp = neuralNet.getLearningRule();
            bp.setMaxError(0.02);
            bp.setLearningRate(lr);

            neuralNet.learn(train);

            calculateAccuracy(neuralNet, test);
            calculateAccuracyNeuroph(neuralNet, test);

            calculateMSE(neuralNet, test);
            calculateMSENeuroph(neuralNet, test);
            numberOfIterations += bp.getCurrentIteration();

        }

        saveNetWithMaxAccuracy();
        System.out.println("Prosecan br iter: " + (double) numberOfIterations / learningRates.length);
    }

    private void saveNetWithMaxAccuracy() {
        Training max = trainings.get(0);
        for (int i = 1; i < trainings.size(); i++) {
            if (max.accuracy < trainings.get(i).accuracy) {
                max = trainings.get(i);
            }
        }

        max.neuralNet.save("nnMax.nnet");
    }

    private void calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {

        ConfussionMatrix cm = new ConfussionMatrix(outputCount);

        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            int actual = getMaxIndex(row.getDesiredOutput());
            int predicted = getMaxIndex(neuralNet.getOutput());

            cm.incrementMatrix(actual, predicted);
        }

        cm.writeMatrix();

        double accuracy = 0;

        for (int i = 0; i < outputCount; i++) {
            accuracy += (double) (cm.getTruePositives(i) + cm.getTrueNegatives(i)) / cm.total;
        }

        double averageAccuracy = (double) accuracy / outputCount;
        Training t = new Training(neuralNet, averageAccuracy);
        trainings.add(t);

        System.out.println("Accuracy: " + averageAccuracy);

    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = {"prvo", "drugo", "trece"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        //evaluation.evaluateDataSet(neuralNet, test);
        evaluation.evaluate(neuralNet, test);
        
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix cm = evaluator.getResult();
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);

        System.out.println("Accuracy njihova metoda: " + average.accuracy);
    }

    private void calculateMSENeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = {"prvo", "drugo", "trece"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));

        //evaluation.evaluateDataSet(neuralNet, test);
        evaluation.evaluate(neuralNet, test);
        System.out.println("Srednja kv greska njihova metoda: " + evaluation.getMeanSquareError());
    }

    private void calculateMSE(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumError = 0, meanErr = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();

            double[] actual = row.getDesiredOutput();
            double[] predicted = neuralNet.getOutput();

            for (int i = 0; i < actual.length; i++) {
                sumError += Math.pow((actual[i] - predicted[i]), 2);
            }
        }

        meanErr = (double) sumError / (2 * test.size());
        System.out.println("Srednja kv greska: " + meanErr);
    }

    private int getMaxIndex(double[] array) {
        int max = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[max]) {
                max = i;
            }
        }
        return max;
    }

}
