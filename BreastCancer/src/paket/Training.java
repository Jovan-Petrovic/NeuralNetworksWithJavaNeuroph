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
    private NeuralNetwork neuralNet;
    private double error;

    public Training() {
    }

    public Training(NeuralNetwork neuralNet, double error) {
        this.neuralNet = neuralNet;
        this.error = error;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }
    
    
}
