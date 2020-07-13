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
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;

/**
 *
 * @author Bron Zilar
 */
public class Glass {
    int inputCount = 9;
    int outputCount = 7;
    double[] learningRates = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {10, 20, 30};
    ArrayList<Training> trainings = new ArrayList<>();
    
    
    public static void main(String[] args) {
        new Glass().run();
    }

    private void run() {
        // 1
        DataSet dataSet = DataSet.createFromFile("glass.csv", inputCount, outputCount, ",");
        
        // 2
        MaxNormalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();
        
        // 3.
        DataSet[] trainAndTest = dataSet.createTrainingAndTestSubsets(0.65, 0.35);
        DataSet train = trainAndTest[0];
        DataSet test = trainAndTest[1];
        
        // 4.
        for(double lr : learningRates) {
            for(int hn : hiddenNeurons) {
                System.out.println("Training neural net with learning rate: " + lr + " and hidden neurons " + hn);
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hn, outputCount);
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                bp.setMomentum(0.6);
                bp.setLearningRate(lr);
                bp.setMaxError(0.2);
                
                neuralNet.learn(train);
                
                double accuracy = calculateAccuracy(neuralNet, test);
                calculateAccuracyNeuroph(neuralNet, test);
                
                Training t = new Training(neuralNet, accuracy, bp.getCurrentIteration());
                trainings.add(t);
                
                calculateMSE(neuralNet, test);
                calculateMSENeuroph(neuralNet, test);
                
            }
        }
        
        calculateAverageIter();
        saveNetWithMaxAcc();
    }

    private double calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        ConfusionMat cm = new ConfusionMat(outputCount);
        
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            int actual = getMaxInArray(row.getDesiredOutput());
            int predicted = getMaxInArray(neuralNet.getOutput());
            
            cm.incrementMatrix(actual, predicted);
        }
        cm.printMatrix();
        
        double accuracy = 0;
        for(int i = 0; i < outputCount; i++) {
            int tp = cm.getTruePositives(i);
            int tn = cm.getTrueNegatives(i);
            int fp = cm.getFalsePositives(i);
            int fn = cm.getFalseNegatives(i);
            
            accuracy += (double) (tp+tn)/(tp+tn+fp+fn);
        }
        double avgAcc = (double) accuracy / outputCount;
        System.out.println("Accuracy: " + avgAcc);
        return avgAcc;
    }

    private void calculateAccuracyNeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = {"s1","s2","s3","s4","s5","s6","s7"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, test);
        
        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
        ConfusionMatrix cm = evaluator.getResult();
        System.out.println("Confusion matrix: ");
        System.out.println(cm.toString());
        ClassificationMetrics[] metricses = ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metricses);
        double accuracy = average.accuracy;
        
        System.out.println("Accuracy - Neuroph: " + accuracy);
    }

    private void calculateMSE(MultiLayerPerceptron neuralNet, DataSet test) {
        double error = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double[] predicted = neuralNet.getOutput();
            double[] actual = row.getDesiredOutput();
            
            for(int i = 0; i < actual.length; i++) {
                error += Math.pow((predicted[i] - actual[i]), 2);
            }
        }
        double mse = error / (2*test.size());
        System.out.println("Mean square error: " + mse);
    }

    private void calculateMSENeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        String[] classLabels = {"s1","s2","s3","s4","s5","s6","s7"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, test);
        
        System.out.println("Mean square error - Neuroph: " + evaluation.getMeanSquareError() + "\n");
    }

    private int getMaxInArray(double[] array) {
        int position = 0;
        for(int i = 1; i < array.length; i++) {
            if(array[position] < array[i]) {
                position = i;
            }
        }
        return position;
    }

    private void calculateAverageIter() {
        int total = 0;
        for(Training t : trainings) {
            total += t.getIterations();
        }
        double avgIterations = (double) total / trainings.size();
        System.out.println("Average number of iterations: " + avgIterations);
    }

    private void saveNetWithMaxAcc() {
        Training max = trainings.get(0);
        for(Training t : trainings) {
            if(t.getAccuracy() > max.getAccuracy()) {
                max = t;
            }
        }
        System.out.println("Neural net with max accuracy: " + max.getAccuracy());
        max.getNeuralNetwork().save("neuralNet.nnet");
    }
}
