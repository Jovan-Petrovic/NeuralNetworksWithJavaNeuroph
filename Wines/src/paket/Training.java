/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author jelica
 */
public class Training {

    NeuralNetwork neuralNet;
    double accuracy;

    public Training() {
    }

    public Training(NeuralNetwork neuralNet, double accuracy) {
        this.neuralNet = neuralNet;
        this.accuracy = accuracy;
    }
    
    
}
