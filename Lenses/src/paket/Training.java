/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;

/**
 *
 * @author Bron Zilar
 */
public class Training {
    NeuralNetwork neuralNetwork;
    DataSet dataset;
    double error;
    int iterations;
    int hiddenNeurons;
    double learningRate;

    public Training(NeuralNetwork neuralNetwork, DataSet dataset, int hiddenNeurons, double learningRate, double error, int iterations ) {
        this.neuralNetwork = neuralNetwork;
        this.dataset = dataset;
        this.error = error;
        this.iterations = iterations;
        this.hiddenNeurons = hiddenNeurons;
        this.learningRate = learningRate; 
    }

    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public DataSet getDataset() {
        return dataset;
    }

    public void setDataset(DataSet dataset) {
        this.dataset = dataset;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public int getHiddenNeurons() {
        return hiddenNeurons;
    }

    public void setHiddenNeurons(int hiddenNeurons) {
        this.hiddenNeurons = hiddenNeurons;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
