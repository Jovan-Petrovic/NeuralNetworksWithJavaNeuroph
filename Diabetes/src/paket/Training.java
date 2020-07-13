/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author Bron Zilar
 */
public class Training {
    private NeuralNetwork neuralNetwork;
    private double accuracy;
    private int numOfIterations;

    public Training() {
    }

    public Training(NeuralNetwork neuralNetwork, double accuracy, int numOfIterations) {
        this.neuralNetwork = neuralNetwork;
        this.accuracy = accuracy;
        this.numOfIterations = numOfIterations;
    }

    public int getNumOfIterations() {
        return numOfIterations;
    }

    public void setNumOfIterations(int numOfIterations) {
        this.numOfIterations = numOfIterations;
    }

    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }
    
    
}
