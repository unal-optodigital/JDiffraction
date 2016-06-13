/*
 * Copyright 2014 Universidad Nacional de Colombia
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package unal.od.jdiffraction.prop;

import org.jtransforms.fft.DoubleFFT_2D;
import unal.od.jdiffraction.utils.DoubleArrayUtils;

/**
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 */
public class DoubleAngularSpectrum extends DoublePropagator {

    private final int M, N;
    private final double z, lambda, dx, dy;
    private final double[][] kernel;
    private final DoubleFFT_2D fft;

    /**
     * Constructor.
     *
     * @param M Number of data points on x direction.
     * @param N Number of data points on y direction.
     * @param lambda Wavelenght
     * @param z Distance.
     * @param dx Sampling pitch on x direction.
     * @param dy Sampling pitch on y direction.
     */
    public DoubleAngularSpectrum(int M, int N, double lambda, double z, double dx, double dy) {
        this.M = M;
        this.N = N;
        this.lambda = lambda;
        this.dx = dx;
        this.dy = dy;
        this.z = z;

        kernel = new double[M][2 * N];
        fft = new DoubleFFT_2D(M, N);

        calculateKernels();
    }

    private void calculateKernels() {

        int M2, N2;
        double kernelFactor, lambdaSq, dfx, dfy, dfxSq, dfySq;

        M2 = M / 2;
        N2 = N / 2;
        lambdaSq = lambda * lambda;
        dfx = 1 / (dx * M);
        dfy = 1 / (dy * N);
        dfxSq = dfx * dfx;
        dfySq = dfy * dfy;
        kernelFactor = (2 * Math.PI * z) / lambda;

        for (int i = 0; i < M; i++) {
            int i2 = i - M2 + 1;
            double c1 = i2 * i2 * dfxSq;

            for (int j = 0; j < N; j++) {
                int j2 = j - N2 + 1;
                double kernelPhase;

                kernelPhase = c1 + j2 * j2 * dfySq;
                kernelPhase *= lambdaSq;
                kernelPhase = 1 - kernelPhase;
                kernelPhase = Math.sqrt(kernelPhase);
                kernelPhase *= kernelFactor;

                kernel[i][2 * j] = Math.cos(kernelPhase);
                kernel[i][2 * j + 1] = Math.sin(kernelPhase);
            }
        }

    }

    @Override
    public void diffract(double[][] field) {

        if (M != field.length || N != (field[0].length / 2)) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        fft.complexForward(field);
        DoubleArrayUtils.complexShift(field);
        DoubleArrayUtils.complexMultiplication2(field, kernel);
        DoubleArrayUtils.complexShift(field);
        fft.complexInverse(field, false);
    }

    public int getM() {
        return M;
    }

    public int getN() {
        return N;
    }

    public double getZ() {
        return z;
    }

    public double getLambda() {
        return lambda;
    }

    public double getDx() {
        return dx;
    }

    public double getDy() {
        return dy;
    }

}
