/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

/**
 *
 * @author Bron Zilar
 */
public class ConfusionMat {
    int total;
    int matrix[][];
    int classCount;

    public ConfusionMat(int classCount) {
        this.total = 0;
        this.matrix = new int[classCount][classCount];
        this.classCount = classCount;
    }

    
    public void incrementMatrix(int actual, int predicted) {
        total++;
        matrix[actual][predicted]++;
    }
    
    public int getTruePositives(int classId) {
        return matrix[classId][classId];
    }
    
    public int getTrueNegatives(int classId) {
        int tn = 0;
        for(int i = 0; i < classCount; i++) {
            if(i == classId) {
                continue;
            }
            for(int j = 0; j < classCount; j++) {
                if(j == classId) {
                    continue;
                }
                tn += matrix[i][j];
            }
        }
        return tn;
    }
    
    public int getFalsePositives(int classId) {
        int fp = 0;
        for(int i = 0; i < classCount; i++) {
            if(i == classId) {
                continue;
            }
            fp += matrix[i][classId];
        }
        return fp;
    }
    
    public int getFalseNegatives(int classId) {
        int fn = 0;
        for(int i = 0; i < classCount; i++) {
            if(i == classId) {
                continue;
            }
            fn += matrix[classId][i];
        }
        return fn;
    } 
    
    public void printMatrix() {
        System.out.println("Confusion matrix: ");
        for (int i = 0; i < classCount; i++) {
            for (int j = 0; j < classCount; j++) {
                System.out.print(matrix[i][j]+" ");
            }
            System.out.println();
        }
    }
}
