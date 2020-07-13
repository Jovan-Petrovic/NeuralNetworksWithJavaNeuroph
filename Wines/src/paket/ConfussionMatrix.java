/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paket;

/**
 *
 * @author jelica
 */
public class ConfussionMatrix {

    int[][] matrix;
    int total;
    int classCount;

    public ConfussionMatrix(int classCount) {
        this.matrix = new int[classCount][classCount];
        this.classCount = classCount;
        this.total = 0;
    }
    
    public void incrementMatrix(int i, int j) {
        total++;
        matrix[i][j]++;
    }

    public int getTruePositives(int klasa) {
        return matrix[klasa][klasa];
    }

    public int getTrueNegatives(int klasa) {
        int tn = 0;
        for (int i = 0; i < classCount; i++) {
            if (i == klasa) {
                continue;
            }
            for (int j = 0; j < classCount; j++) {
                if (j == klasa) {
                    continue;
                }
                tn += matrix[i][j];
            }
        }
        return tn;
    }

    public int getFalsePositives(int klasa) {
        int fp = 0;
        for (int i = 0; i < classCount; i++) {
            if(i == klasa) continue;
            fp += matrix[i][klasa];
        }
        return fp;
    }

    public int getFalseNegatives(int klasa) {
        int fn = 0;
        for (int i = 0; i < classCount; i++) {
            if(i == klasa) continue;
            fn += matrix[klasa][i];
        }
        return fn;
    }

    public void writeMatrix() {
        System.out.println("Matrica konfuzije");
        for (int i = 0; i < classCount; i++) {
            for (int j = 0; j < classCount; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

}
