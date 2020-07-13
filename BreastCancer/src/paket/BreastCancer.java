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
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Bron Zilar
 */
public class BreastCancer {
    int inputCount = 30;
    int outputCount = 1;
    double[] learningRates = {0.2, 0.4, 0.6};
    int[] hiddenNeurons = {10, 20, 30};
    ArrayList<Training> trainings = new ArrayList<>();
    
    public static void main(String[] args) {
        new BreastCancer().run();
    }

    private void run() {
        // 1.
        DataSet dataSet = DataSet.createFromFile("breast_cancer_data.csv", inputCount, outputCount, ",");
        
        // 2.
        Normalizer normalizer = new MaxNormalizer(dataSet);
        normalizer.normalize(dataSet);
        dataSet.shuffle();
        
        // 3.
        DataSet[] trainAndtest = dataSet.createTrainingAndTestSubsets(0.65, 0.35);
        DataSet train = trainAndtest[0];
        DataSet test = trainAndtest[1];
        
        // 4.
        int numOfIterations = 0, numOfTrainings = 0;
        
        for (double lr : learningRates) {
            for (int hn : hiddenNeurons) {
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hn, outputCount);
                neuralNet.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation bp = (MomentumBackpropagation) neuralNet.getLearningRule();
                bp.setMomentum(0.7);
                bp.setLearningRate(lr);
                bp.setMaxError(0.07);
                
                neuralNet.learn(train);
                
                calculateMSE(neuralNet, test);
                calculateMSENeuroph(neuralNet, test);
                
                calculateAccuracy(neuralNet, test);
                calculateAccuracyNeuroph(neuralNet, test);
                
                numOfIterations += bp.getCurrentIteration();
                numOfTrainings++;
                
            }
        }
        
        System.out.println("Mean number of iterations: " + (double) numOfIterations/numOfTrainings);
        saveNetWithMinError();
    }

    private void calculateMSE(MultiLayerPerceptron neuralNet, DataSet test) {
        double sumError = 0, mse = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            sumError += Math.pow((predicted - actual), 2);
        }
        mse = (double) sumError / (2*test.size());
        System.out.println("Mean squared error: " + mse);
        
        Training t = new Training(neuralNet, mse);
        trainings.add(t);
    }

    private void calculateMSENeuroph(MultiLayerPerceptron neuralNet, DataSet test) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, test);
        
        System.out.println("Mean squared error - Neuroph: " + evaluation.getMeanSquareError());
    }

    private void calculateAccuracy(MultiLayerPerceptron neuralNet, DataSet test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (DataSetRow row : test) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate();
            
            double predicted = neuralNet.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            if(actual == 1 && predicted > 0.5) tp++;
            if(actual == 1 && predicted <= 0.5) fn++;
            if(actual == 0 && predicted > 0.5) fp++;
            if(actual == 0 && predicted <= 0.5) tn++;
        }
        double accuracy = (double) (tp+tn) / (tp+tn+fp+fn);
        System.out.println("Accuracy: " + accuracy);
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
        
        System.out.println("Accuracy - Neuroph: " + accuracy + "\n");
    }

    private void saveNetWithMinError() {
        Training min = trainings.get(0);
        for(Training t : trainings) {
            if(t.getError() < min.getError()) {
                min = t;
            }
        }
        min.getNeuralNet().save("neuralNet.nnet");
    }
}
