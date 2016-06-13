/*
 * Copyright 2016 Universidad Nacional de Colombia
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
package unal.od.jdiffraction.cpu;

import org.jtransforms.fft.FloatFFT_2D;
import unal.od.jdiffraction.cpu.utils.ArrayUtils;

/**
 * Computes wave diffraction through angular spectrum method with single
 * precision.
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.0
 */
public class FloatAngularSpectrum extends FloatPropagator {

    private final int M, N;
    private final float z, lambda, dx, dy;
    private final float[][] kernel;
    private final FloatFFT_2D fft;

    /**
     * Creates a new instance of FloatAngularSpectrum. Also performs kernel
     * calculations.
     *
     * @param M Number of data points on x direction.
     * @param N Number of data points on y direction.
     * @param lambda Wavelength.
     * @param z Distance.
     * @param dx Sampling pitch on x direction.
     * @param dy Sampling pitch on y direction.
     */
    public FloatAngularSpectrum(int M, int N, float lambda, float z, float dx, float dy) {
        this.M = M;
        this.N = N;
        this.lambda = lambda;
        this.dx = dx;
        this.dy = dy;
        this.z = z;

        kernel = new float[M][2 * N];
        fft = new FloatFFT_2D(M, N);

        calculateKernels();
    }

    private void calculateKernels() {

        int M2, N2, endM, endN;
        float kernelFactor, lambdaSq, dfx, dfy, dfxSq, dfySq;

        M2 = M / 2;
        N2 = N / 2;
        endM = 2 * M2 - 1;
        endN = 2 * N2 - 1;
        lambdaSq = lambda * lambda;
        dfx = 1 / (dx * M);
        dfy = 1 / (dy * N);
        dfxSq = dfx * dfx;
        dfySq = dfy * dfy;
        kernelFactor = (2 * (float) Math.PI * z) / lambda;

        for (int i = 0; i < M2; i++) {
            int i2 = i - M2 + 1;
            float c1 = i2 * i2 * dfxSq;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                float kernelPhase;

                kernelPhase = c1 + j2 * j2 * dfySq;
                kernelPhase *= lambdaSq;
                kernelPhase = 1 - kernelPhase;
                kernelPhase = (float) Math.sqrt(kernelPhase);
                kernelPhase *= kernelFactor;

                kernel[i][2 * j] = kernel[endM - i][2 * j] = kernel[i][2 * (endN - j)]
                        = kernel[endM - i][2 * (endN - j)] = (float) Math.cos(kernelPhase);
                kernel[i][2 * j + 1] = kernel[endM - i][2 * j + 1] = kernel[i][2 * (endN - j) + 1]
                        = kernel[endM - i][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase);
            }
        }

        if (M % 2 != 0) {
            int i2 = M - M2 + 1;
            float c1 = i2 * i2 * dfxSq;

            for (int j = 0; j < N2; j++) {
                int j2 = j - N2 + 1;
                float kernelPhase;

                kernelPhase = c1 + j2 * j2 * dfySq;
                kernelPhase *= lambdaSq;
                kernelPhase = 1 - kernelPhase;
                kernelPhase = (float) Math.sqrt(kernelPhase);
                kernelPhase *= kernelFactor;

                kernel[M - 1][2 * j] = kernel[M - 1][2 * (endN - j)] = (float) Math.cos(kernelPhase);
                kernel[M - 1][2 * j + 1] = kernel[M - 1][2 * (endN - j) + 1] = (float) Math.sin(kernelPhase);
            }
        }

        if (N % 2 != 0) {
            int j2 = N - N2 + 1;
            float c1 = j2 * j2 * dfySq;

            for (int i = 0; i < N2; i++) {
                int i2 = M - M2 + 1;
                float kernelPhase;

                kernelPhase = c1 + i2 * i2 * dfxSq;
                kernelPhase *= lambdaSq;
                kernelPhase = 1 - kernelPhase;
                kernelPhase = (float) Math.sqrt(kernelPhase);
                kernelPhase *= kernelFactor;

                kernel[i][2 * (N - 1)] = kernel[endM - i][2 * (N - 1)] = (float) Math.cos(kernelPhase);
                kernel[i][2 * (N - 1) + 1] = kernel[endM - i][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase);
            }
        }

        if (M % 2 != 0 && N % 2 != 0) {
            int i2 = M - M2 + 1;
            int j2 = N - N2 + 1;

            float kernelPhase;

            kernelPhase = i2 * i2 * dfxSq + j2 * j2 * dfySq;
            kernelPhase *= lambdaSq;
            kernelPhase = 1 - kernelPhase;
            kernelPhase = (float) Math.sqrt(kernelPhase);
            kernelPhase *= kernelFactor;

            kernel[M - 1][2 * (N - 1)] = (float) Math.cos(kernelPhase);
            kernel[M - 1][2 * (N - 1) + 1] = (float) Math.sin(kernelPhase);
        }
    }

    @Override
    public void diffract(float[][] field) {
        if (M != field.length || 2 * N != field[0].length) {
            throw new IllegalArgumentException("Array dimension must be " + M + " x " + 2 * N + ".");
        }

        fft.complexForward(field);
        ArrayUtils.complexShift(field);
        ArrayUtils.complexMultiplication2(field, kernel);
        ArrayUtils.complexShift(field);
        fft.complexInverse(field, true);
    }

    public int getM() {
        return M;
    }

    public int getN() {
        return N;
    }

    public float getZ() {
        return z;
    }

    public float getLambda() {
        return lambda;
    }

    public float getDx() {
        return dx;
    }

    public float getDy() {
        return dy;
    }

}
