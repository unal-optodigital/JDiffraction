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
package unal.od.jdiffraction.utils;

/**
 * Utilities for working with complex and real data arrays. Complex arrays
 * layout has to be as follows:
 * <p>
 *
 * <pre>
 * a[i][2*j] = Re[i][j]
 * a[i][2*j+1] = Im[i][j]; 0&lt;=i&lt;M, 0&lt;=j&lt;N
 * </pre>
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 */
public class DoubleArrayUtils {

    private DoubleArrayUtils() {
    }

    private static void checkDimension(double[][] a) {
        if (a.length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        } else if (a[0].length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        }
    }

    /**
     * Computes the phase (angle) of a complex array
     *
     * @param a complex array
     * @return phase (angle) array
     */
    public static double[][] phase(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] phase = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                phase[i][j] = Math.atan2(a[i][2 * j + 1], a[i][2 * j]);
            }
        }
        return phase;
    }

    /**
     * Computes the modulus of a complex array
     *
     * @param a complex array
     * @return modulus array
     */
    public static double[][] modulus(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] modulus = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                modulus[i][j] = a[i][2 * j] * a[i][2 * j];
                modulus[i][j] += a[i][2 * j + 1] * a[i][2 * j + 1];
                modulus[i][j] = Math.sqrt(modulus[i][j]);
            }
        }
        return modulus;
    }

    /**
     * Computes the squared modulus of a complex array
     *
     * @param a complex array
     * @return modulus squared array
     */
    public static double[][] modulusSq(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] modulusSq = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                modulusSq[i][j] = (a[i][2 * j] * a[i][2 * j]) + (a[i][2 * j + 1] * a[i][2 * j + 1]);
            }
        }
        return modulusSq;
    }

    /**
     * Computes the element by element complex multiplication of 2 arrays
     *
     * @param a complex array
     * @param b complex array
     * @return multiplication
     */
    public static double[][] complexMultiplication(double[][] a, double[][] b) {
        checkDimension(a);
        checkDimension(b);
        int M = a.length;
        int N = a[0].length;
        if (M != b.length || N != b[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        double[][] multiplied = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j++) {
                multiplied[i][2 * j] = (a[i][2 * j] * b[i][2 * j]) - (a[i][2 * j + 1] * b[i][2 * j + 1]);
                multiplied[i][2 * j + 1] = (a[i][2 * j] * b[i][2 * j + 1]) + (a[i][2 * j + 1] * b[i][2 * j]);
            }
        }
        return multiplied;
    }

    /**
     * Computes the element by element complex multiplication of 2 arrays
     * leaving the result in <code>a</code>
     *
     * @param a complex array
     * @param b complex array
     */
    public static void complexMultiplication2(double[][] a, double[][] b) {
        checkDimension(a);
        checkDimension(b);
        int M = a.length;
        int N = a[0].length;
        if (M != b.length || N != b[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j++) {
                double real = a[i][2 * j];
                double imaginary = a[i][2 * j + 1];

                a[i][2 * j] = (real * b[i][2 * j]) - (imaginary * b[i][2 * j + 1]);
                a[i][2 * j + 1] = (real * b[i][2 * j + 1]) + (imaginary * b[i][2 * j]);
            }
        }
    }

    /**
     * Computes a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param phase phase array
     * @param amp amplitude array
     * @return complex array
     */
    public static double[][] complexAmplitude(double[][] phase, double[][] amp) {
        checkDimension(phase);
        checkDimension(amp);
        int M = phase.length;
        int N = phase[0].length;
        if (M != amp.length || N != amp[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        double[][] complexAmp = new double[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp[i][j] * Math.cos(phase[i][j]);
                complexAmp[i][2 * j + 1] = amp[i][j] * Math.sin(phase[i][j]);
            }
        }
        return complexAmp;
    }

    /**
     * Computes a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param phase phase array
     * @param amp amplitude
     * @return complex array
     */
    public static double[][] complexAmplitude(double[][] phase, double amp) {
        checkDimension(phase);
        int M = phase.length;
        int N = phase[0].length;

        double[][] complexAmp = new double[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp * Math.cos(phase[i][j]);
                complexAmp[i][2 * j + 1] = amp * Math.sin(phase[i][j]);
            }
        }
        return complexAmp;
    }

    /**
     * Computes a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param phase phase
     * @param amp amplitude array
     * @return complex array
     */
    public static double[][] complexAmplitude(double phase, double[][] amp) {
        checkDimension(amp);
        int M = amp.length;
        int N = amp[0].length;

        double[][] complexAmp = new double[M][2 * N];
        double cos = Math.cos(phase);
        double sin = Math.sin(phase);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp[i][j] * cos;
                complexAmp[i][2 * j + 1] = amp[i][j] * sin;
            }
        }
        return complexAmp;
    }

    /**
     * Computes a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param M X dimension
     * @param N Y dimension
     * @param phase phase
     * @param amp amplitude
     * @return complex array
     */
    public static double[][] complexAmplitude(int M, int N, double phase, double amp) {
        if (M == 0 || N == 0) {
            throw new IllegalArgumentException("Dimensions must be greater than 0.");
        }

        double[][] complexAmp = new double[M][2 * N];
        double aCos = amp * Math.cos(phase);
        double aSin = amp * Math.sin(phase);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = aCos;
                complexAmp[i][2 * j + 1] = aSin;
            }
        }
        return complexAmp;
    }

    /**
     * Takes the real and imaginary parts and returns a complex array
     *
     * @param real real part array
     * @param imaginary imaginary part array
     * @return complex array
     */
    public static double[][] complexAmplitude2(double[][] real, double[][] imaginary) {
        checkDimension(real);
        checkDimension(imaginary);
        int M = real.length;
        int N = real[0].length;
        if (M != imaginary.length || N != imaginary[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        double[][] complexAmp = new double[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = real[i][j];
                complexAmp[i][2 * j + 1] = imaginary[i][j];
            }
        }
        return complexAmp;
    }

    /**
     * Takes the real part (imaginary = 0) and returns a complex array
     *
     * @param real real part array
     * @return complex array
     */
    public static double[][] complexAmplitude2(double[][] real) {
        checkDimension(real);
        int M = real.length;
        int N = real[0].length;

        double[][] complexAmp = new double[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = real[i][j];
                complexAmp[i][2 * j + 1] = 0;
            }
        }
        return complexAmp;
    }

    /**
     * Computes a complex array pointwise leaving the result in <code>a</code>.
     * The computation is done calculating <code>amp * exp(i * phase(a))</code>.
     *
     * @param a complex array
     * @param amp amplitude array
     */
    public static void complexAmplitude3(double[][] a, double[][] amp) {
        checkDimension(a);
        checkDimension(amp);
        int M = a.length;
        int N = a[0].length / 2;
        if (M != amp.length || N != amp[0].length) {
            throw new IllegalArgumentException("Amplitude array dimension must be " + M + " x " + N + ".");
        }

        double[][] tmp = phase(a);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][2 * j] = amp[i][j] * Math.cos(tmp[i][j]);
                a[i][2 * j + 1] = amp[i][j] * Math.sin(tmp[i][j]);
            }
        }
    }

    /**
     * Computes a complex array pointwise leaving the result in <code>a</code>.
     * The computation is done calculating <code>amp * exp(i * phase(a))</code>.
     *
     * @param a complex array
     * @param amp amplitude
     */
    public static void complexAmplitude3(double[][] a, double amp) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] tmp = phase(a);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][2 * j] = amp * Math.cos(tmp[i][j]);
                a[i][2 * j + 1] = amp * Math.sin(tmp[i][j]);
            }
        }
    }

    /**
     * Extracts the real part of a complex array
     *
     * @param a complex array
     * @return real array
     */
    public static double[][] real(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] real = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                real[i][j] = a[i][2 * j];
            }
        }

        return real;
    }

    /**
     * Extracts the imaginary part of a complex array
     *
     * @param a complex array
     * @return imaginary array
     */
    public static double[][] imaginary(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        double[][] imaginary = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                imaginary[i][j] = a[i][2 * j + 1];
            }
        }

        return imaginary;
    }

    /**
     * Performs the circular shifting of a complex array, leaving the result in
     * <code>a</code>
     *
     * @param a complex array
     */
    public static void complexShift(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        int M2 = M / 2;
        int N2 = N / 2;

        double tmp;

        for (int i = 0; i < M2; i++) {
            for (int j = 0; j < N2; j++) {
                //Real shift
                tmp = a[i][2 * j];
                a[i][2 * j] = a[i + M2][2 * (j + N2)];
                a[i + M2][2 * (j + N2)] = tmp;

                tmp = a[i + M2][2 * j];
                a[i + M2][2 * j] = a[i][2 * (j + N2)];
                a[i][2 * (j + N2)] = tmp;

                //Imag shift
                tmp = a[i][2 * j + 1];
                a[i][2 * j + 1] = a[i + M2][(2 * (j + N2)) + 1];
                a[i + M2][(2 * (j + N2)) + 1] = tmp;

                tmp = a[i + M2][2 * j + 1];
                a[i + M2][2 * j + 1] = a[i][2 * (j + N2) + 1];
                a[i][2 * (j + N2) + 1] = tmp;
            }
        }
    }

    /**
     * Performs the circular shifting of a real array, leaving the result in
     * <code>a</code>
     *
     * @param a array
     */
    public static void realShift(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        int M2 = M / 2;
        int N2 = N / 2;

        double tempShift;

        for (int i = 0; i < M2; i++) {
            for (int j = 0; j < N2; j++) {
                tempShift = a[i][j];
                a[i][j] = a[i + M2][j + N2];
                a[i + M2][j + N2] = tempShift;

                tempShift = a[i + M2][j];
                a[i + M2][j] = a[i][j + N2];
                a[i][j + N2] = tempShift;
            }
        }
    }

    /**
     * Computes log10 of a real array
     *
     * @param a array
     * @return array containing log10(a)
     */
    public static double[][] log10(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        double[][] b = new double[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                b[i][j] = Math.log10(a[i][j]);
            }
        }
        return b;
    }

    /**
     * Gets the max value of a real array
     *
     * @param a array
     * @return max
     */
    public static double max(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        double max = a[0][0];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                max = Math.max(max, a[i][j]);
            }
        }
        return max;
    }

    /**
     * Gets the min value of a real array
     *
     * @param a array
     * @return min
     */
    public static double min(double[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;
        double min = a[0][0];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                min = Math.min(min, a[i][j]);
            }
        }
        return min;
    }

    /**
     *
     * @param a
     * @param max
     * @param min
     * @param maxScale
     * @return
     */
    public static double[][] scale(double[][] a, double max, double min, double maxScale) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        double[][] scaled = new double[M][N];

        double delta = max - min;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                scaled[i][j] = a[i][j] - min;
                scaled[i][j] = scaled[i][j] / delta;
                scaled[i][j] = scaled[i][j] * maxScale;
            }
        }
        return scaled;
    }

    /**
     *
     * @param a
     * @param maxScale
     * @return
     */
    public static double[][] scale(double[][] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        return scale(a, max, min, maxScale);
    }

    /**
     *
     * @param a
     * @param max
     * @param min
     * @param maxScale
     */
    public static void scale2(double[][] a, double max, double min, double maxScale) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        double delta = max - min;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] - min;
                a[i][j] = a[i][j] / delta;
                a[i][j] = a[i][j] * maxScale;
            }
        }
    }

    /**
     *
     * @param a
     * @param maxScale
     */
    public static void scale2(double[][] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        scale2(a, max, min, maxScale);
    }
}
