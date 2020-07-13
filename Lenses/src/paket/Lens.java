/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

import java.util.ArrayList;
import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

/**
 *
 * @author Bron Zilar
 */
public class Lens {
    
    int inputCount = 9;
    int outputCount = 3;
    
    double minLearningRate = 0.1;
    double maxLearningRate = 0.9;
        
    int minHiddenNeurons = 10;
    int maxHiddenNeurons = 20;
    
    ArrayList<Training> trainings = new ArrayList<>();
    Training minErrorTraining, minIterationsTraining;
    int trainingCount=0;
    
    public static void main(String[] args) {
        new Lens().run();
    }

    private void run() {
        DataSet dataSet = DataSet.createFromFile("lenses_data.txt", inputCount, outputCount, " ");
        
        for(double lr = minLearningRate; lr <= maxLearningRate; lr+=0.1) {
            for(int hn = minHiddenNeurons; hn <= maxHiddenNeurons; hn++) {
                
                MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hn, outputCount);
                BackPropagation bp = neuralNet.getLearningRule();
                bp.setLearningRate(lr);
                bp.setMaxIterations(20000);
                bp.setMaxError(0.01);
                
                bp.addListener((event) -> {
                    BackPropagation backPropagation = (BackPropagation) event.getSource();
                    System.out.println(backPropagation.getCurrentIteration() + ". iteratin | Total Network Error: " + backPropagation.getTotalNetworkError());
                });
                 
                trainingCount++;
                System.out.println("-------------------------------------------------------------------");
                System.out.println("Training #"+trainingCount);
                System.out.println("Hidden neurons "+ hn + " Learning rate: "+ lr +"\n");
                
                neuralNet.learn(dataSet);
                
                int iterations = bp.getCurrentIteration();
                double totalError = bp.getTotalNetworkError();
                
                Training t = new Training(neuralNet, dataSet, hn, lr, totalError, iterations);
                trainings.add(t);

            }
        }
        
        analyzeTrainings();
        
        // najmanji broj gresaka i iteracija koje smo izracunali u ovoj metodi analyze
        System.out.println("Testing network with min error");
        testNeuralNetwork(minErrorTraining.getNeuralNetwork(), minErrorTraining.getDataset());
        System.out.println("Testing network with min iterations");
        testNeuralNetwork(minIterationsTraining.getNeuralNetwork(), minIterationsTraining.getDataset());
    }

    private void analyzeTrainings() {
        double avgError, avgIterations; 
        double errorSum = 0, iterationsSum = 0;
        
        minErrorTraining = trainings.get(0);
        minIterationsTraining = trainings.get(0);
        
        for(Training training : trainings) {
            if (training.getIterations() < minIterationsTraining.getIterations()) {
                minIterationsTraining = training;
            }
            
            if (training.getError()< minErrorTraining.getError()) {
                minErrorTraining = training;
            }
            
            errorSum = errorSum + training.getError();
            iterationsSum = iterationsSum + training.getIterations();            
        }
        
        avgError = errorSum / trainings.size();
        avgIterations = iterationsSum / trainings.size();
        
        System.out.println("Average error: "+avgError);
        System.out.println("Average iterations: "+avgIterations);
    }

    private void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {
        for (DataSetRow row : testSet.getRows()) {
            neuralNet.setInput(row.getInput());
            neuralNet.calculate(); // ovo nam treba samo da bismo mogli da dobijemo outpute
            double[] networkOutput = neuralNet.getOutput(); // uzimamo outpute iz mreze

            System.out.print("Input: " + Arrays.toString(row.getInput()));
            System.out.println(" Output: " + Arrays.toString(networkOutput));
            // mora to string da se napise da bi ispisivao stringove
        }
    }

    
}
