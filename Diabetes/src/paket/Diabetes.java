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
import org.neuroph.eval.Evaluator;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Bron Zilar
 */
public class Diabetes {
    int inputCount = 8;
    int outputCount = 1;
    double[] learningRates = {0.2, 0.4, 0.6};
    ArrayList<Training> trainings = new ArrayList<>();
    
    public static void main(String[] args) {
        new Diabetes().run();
    }

    private void run() {
        // 1
        DataSet dataSet = DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
        
        // 2
        Normalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();
        
        // 3
        DataSet[] trainAndTest = dataSet.createTrainingAndTestSubsets(0.7, 0.3);
        DataSet train = trainAndTest[0];
        DataSet test = trainAndTest[1];
        
        // 4
        for(double lr : learningRates) {
            MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, 20, 10, outputCount);
            BackPropagation  bp = neuralNet.getLearningRule();
            bp.setMaxError(0.07);
            bp.setLearningRate(lr);
            
            neuralNet.learn(train);
            
            double accuracy = calculateAccuracy(neuralNet, test);
            calculateAccuracyNeuroph(neuralNet, test);
            
            calculateMSE(neuralNet, test);
            calculateMSENeuroph(neuralNet, test);
            
            Training t = new Training(neuralNet, accuracy, bp.getCurrentIteration());
            trainings.add(t);
            
        }
        printAverageNumberOfIterations();
        saveNetWithMaxAccuracy();
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            if (actual == 1.0 && predicted > 0.5) {
                tp++;
            }
            if (actual == 0.0 && predicted <= 0.5) {
                tn++;
            }
            if (actual == 1.0 && predicted <= 0.5) {
                fn++;
            }

            if (actual == 0.0 && predicted > 0.5) {
                fp++;
            }
        }
        double accuracy = (double) (tp+tn)/(tp+tn+fp+fn);
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);
        
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.Binary.class);
        ConfusionMatrix cm = evaluator.getResult();
        ClassificationMetrics[] metricses = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metricses);
        
        double accuracy = average.accuracy;
        System.out.println("Accuracy - Neuroph: " + accuracy);
    }

    private void calculateMSE(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumError = 0, mse = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            sumError += Math.pow((predicted-actual), 2);
        }
        mse = (double) sumError / (2*test.size());
        System.out.println("Mean square error: " + mse);
    }

    private void calculateMSENeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);
        
        System.out.println("Mean square error - Neuroph: " + evaluation.getMeanSquareError() + "\n");
        
    }

    private void printAverageNumberOfIterations() {
        int total = 0;
        for(Training t : trainings) {
            total += t.getNumOfIterations();
        }
        System.out.println("Average number of iterations: " + (double) total / trainings.size());
    }

    private void saveNetWithMaxAccuracy() {
        Training max = trainings.get(0);
        for(Training t : trainings) {
            if(t.getAccuracy() > max.getAccuracy()) {
                max = t;
            }
        }
        System.out.println("Max accuracy: " + max.getAccuracy());
        max.getNeuralNetwork().save("neuralNet.nnet");
    }
}
