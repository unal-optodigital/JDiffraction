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
package unal.od.jdiffraction.cpu.utils;

/**
 * Utilities for working with complex and real data arrays. The physical layout
 * of the complex data must be the same as in JTransforms:
 * <p>
 * {@code
 * a[i][2 * j] = Re[i][j],
 * a[i][2 * j + 1] = Im[i][j]; 0 &lt;= i &lt; M, 0 &lt;= j &lt; N
 * }
 *
 * @author Pablo Piedrahita-Quintero (jppiedrahitaq@unal.edu.co)
 * @author Carlos Trujillo (catrujila@unal.edu.co)
 * @author Jorge Garcia-Sucerquia (jigarcia@unal.edu.co)
 *
 * @since JDiffraction 1.1
 */
public class ArrayUtils {

    private static final String VERSION = "1.2";

    private ArrayUtils() {
    }

    private static void checkDimension(float[][] a) {
        if (a.length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        } else if (a[0].length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        }
    }

    private static void checkDimension(double[][] a) {
        if (a.length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        } else if (a[0].length == 0) {
            throw new IllegalArgumentException("Arrays dimension must be greater than 0.");
        }
    }

    /**
     * Returns the library version as a String.
     *
     * @return library version
     *
     * @since JDiffraction 1.2
     */
    public static String jDiffractionVersion() {
        return VERSION;
    }

    /**
     * Computes the phase (angle) of a complex array.
     * <p>
     * {@code
     * phase[i][j] = atan(Im[i][j] / Re[i][j])
     * }
     *
     * @param a complex array
     * @return phase (angle) array
     */
    public static float[][] phase(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        float[][] phase = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                phase[i][j] = (float) Math.atan2(a[i][2 * j + 1], a[i][2 * j]);
            }
        }
        return phase;
    }

    /**
     * Computes the phase (angle) of a complex array.
     * <p>
     * {@code
     * phase[i][j] = atan(Im[i][j] / Re[i][j])
     * }
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
     * Computes the modulus of a complex array.
     * <p>
     * {@code
     * modulus[i][j] = sqrt(Re[i][j]^2 + Im[i][j]^2)
     * }
     *
     * @param a complex array
     * @return modulus array
     */
    public static float[][] modulus(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        float[][] modulus = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                modulus[i][j] = a[i][2 * j] * a[i][2 * j];
                modulus[i][j] += a[i][2 * j + 1] * a[i][2 * j + 1];
                modulus[i][j] = (float) Math.sqrt(modulus[i][j]);
            }
        }
        return modulus;
    }

    /**
     * Computes the modulus of a complex array.
     * <p>
     * {@code
     * modulus[i][j] = sqrt(Re[i][j]^2 + Im[i][j]^2)
     * }
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
     * Computes the squared modulus of a complex array.
     * <p>
     * {@code
     * modulusSq[i][j] = Re[i][j]^2 + Im[i][j]^2
     * }
     *
     * @param a complex array
     * @return modulus squared array
     */
    public static float[][] modulusSq(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        float[][] modulusSq = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                modulusSq[i][j] = (a[i][2 * j] * a[i][2 * j]) + (a[i][2 * j + 1] * a[i][2 * j + 1]);
            }
        }
        return modulusSq;
    }

    /**
     * Computes the squared modulus of a complex array.
     * <p>
     * {@code
     * modulusSq[i][j] = Re[i][j]^2 + Im[i][j]^2
     * }
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
     * Computes the pointwise complex multiplication of 2 arrays.
     *
     * @param a complex array
     * @param b complex array
     * @return multiplication
     */
    public static float[][] complexMultiplication(float[][] a, float[][] b) {
        checkDimension(a);
        checkDimension(b);
        int M = a.length;
        int N = a[0].length;
        if (M != b.length || N != b[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        float[][] multiplied = new float[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j++) {
                multiplied[i][2 * j] = (a[i][2 * j] * b[i][2 * j]) - (a[i][2 * j + 1] * b[i][2 * j + 1]);
                multiplied[i][2 * j + 1] = (a[i][2 * j] * b[i][2 * j + 1]) + (a[i][2 * j + 1] * b[i][2 * j]);
            }
        }
        return multiplied;
    }

    /**
     * Computes the pointwise complex multiplication of 2 arrays.
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

        double[][] multiplied = new double[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j++) {
                multiplied[i][2 * j] = (a[i][2 * j] * b[i][2 * j]) - (a[i][2 * j + 1] * b[i][2 * j + 1]);
                multiplied[i][2 * j + 1] = (a[i][2 * j] * b[i][2 * j + 1]) + (a[i][2 * j + 1] * b[i][2 * j]);
            }
        }
        return multiplied;
    }

    /**
     * Computes the pointwise complex multiplication of 2 arrays leaving the
     * result in {@code a}.
     *
     * @param a complex array
     * @param b complex array
     */
    public static void complexMultiplication2(float[][] a, float[][] b) {
        checkDimension(a);
        checkDimension(b);
        int M = a.length;
        int N = a[0].length;
        if (M != b.length || N != b[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N / 2; j++) {
                float real = a[i][2 * j];
                float imaginary = a[i][2 * j + 1];

                a[i][2 * j] = (real * b[i][2 * j]) - (imaginary * b[i][2 * j + 1]);
                a[i][2 * j + 1] = (real * b[i][2 * j + 1]) + (imaginary * b[i][2 * j]);
            }
        }
    }

    /**
     * Computes the pointwise complex multiplication of 2 arrays leaving the
     * result in {@code a}.
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
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param phase phase array
     * @param amp amplitude array
     * @return complex array
     */
    public static float[][] complexAmplitude(float[][] phase, float[][] amp) {
        checkDimension(phase);
        checkDimension(amp);
        int M = phase.length;
        int N = phase[0].length;
        if (M != amp.length || N != amp[0].length) {
            throw new IllegalArgumentException("Arrays must be equal-sized.");
        }

        float[][] complexAmp = new float[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp[i][j] * (float) Math.cos(phase[i][j]);
                complexAmp[i][2 * j + 1] = amp[i][j] * (float) Math.sin(phase[i][j]);
            }
        }
        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
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
     * Creates a complex array element by element. The computation is done
     * calculating <code>amp * exp(i * phase)</code>
     *
     * @param phase phase array
     * @param amp amplitude
     * @return complex array
     */
    public static float[][] complexAmplitude(float[][] phase, float amp) {
        checkDimension(phase);
        int M = phase.length;
        int N = phase[0].length;

        float[][] complexAmp = new float[M][2 * N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp * (float) Math.cos(phase[i][j]);
                complexAmp[i][2 * j + 1] = amp * (float) Math.sin(phase[i][j]);
            }
        }
        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
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
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
     *
     * @param phase phase
     * @param amp amplitude array
     * @return complex array
     */
    public static float[][] complexAmplitude(float phase, float[][] amp) {
        checkDimension(amp);
        int M = amp.length;
        int N = amp[0].length;

        float[][] complexAmp = new float[M][2 * N];
        float cos = (float) Math.cos(phase);
        float sin = (float) Math.sin(phase);

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                complexAmp[i][2 * j] = amp[i][j] * cos;
                complexAmp[i][2 * j + 1] = amp[i][j] * sin;
            }
        }
        return complexAmp;
    }

    /**
     * Creates a complex array pointwise. The computation is done calculating
     * {@code amp * exp(i * phase)}.
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
     * Takes the real and imaginary parts and returns a complex array. If one of
     * the input arrays is null, that part is filled with zeros.
     *
     * @param real real part array
     * @param imaginary imaginary part array
     * @return complex array
     */
    public static float[][] complexAmplitude2(float[][] real, float[][] imaginary) {
        boolean hasReal = real != null;
        boolean hasImaginary = imaginary != null;

        float[][] complexAmp = null;

        if (!hasReal && !hasImaginary) {
            throw new IllegalArgumentException("Both arguments can't be null.");
        } else if (hasReal && hasImaginary) {
            checkDimension(real);
            checkDimension(imaginary);

            int M = real.length;
            int N = real[0].length;
            if (M != imaginary.length || N != imaginary[0].length) {
                throw new IllegalArgumentException("Arrays must be equal-sized.");
            }

            complexAmp = new float[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = real[i][j];
                    complexAmp[i][2 * j + 1] = imaginary[i][j];
                }
            }
        } else if (hasReal && !hasImaginary) {
            checkDimension(real);

            int M = real.length;
            int N = real[0].length;

            complexAmp = new float[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = real[i][j];
                    complexAmp[i][2 * j + 1] = 0;
                }
            }
        } else if (!hasReal && hasImaginary) {
            checkDimension(imaginary);

            int M = imaginary.length;
            int N = imaginary[0].length;

            complexAmp = new float[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = 0;
                    complexAmp[i][2 * j + 1] = imaginary[i][j];
                }
            }
        }

        return complexAmp;
    }

    /**
     * Takes the real and imaginary parts and returns a complex array. If one of
     * the input arrays is null, that part is filled with zeros.
     *
     * @param real real part array
     * @param imaginary imaginary part array
     * @return complex array
     */
    public static double[][] complexAmplitude2(double[][] real, double[][] imaginary) {
        boolean hasReal = real != null;
        boolean hasImaginary = imaginary != null;

        double[][] complexAmp = null;

        if (!hasReal && !hasImaginary) {
            throw new IllegalArgumentException("Both arguments can't be null.");
        } else if (hasReal && hasImaginary) {
            checkDimension(real);
            checkDimension(imaginary);

            int M = real.length;
            int N = real[0].length;
            if (M != imaginary.length || N != imaginary[0].length) {
                throw new IllegalArgumentException("Arrays must be equal-sized.");
            }

            complexAmp = new double[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = real[i][j];
                    complexAmp[i][2 * j + 1] = imaginary[i][j];
                }
            }
        } else if (hasReal && !hasImaginary) {
            checkDimension(real);

            int M = real.length;
            int N = real[0].length;

            complexAmp = new double[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = real[i][j];
                    complexAmp[i][2 * j + 1] = 0;
                }
            }
        } else if (!hasReal && hasImaginary) {
            checkDimension(imaginary);

            int M = imaginary.length;
            int N = imaginary[0].length;

            complexAmp = new double[M][2 * N];

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    complexAmp[i][2 * j] = 0;
                    complexAmp[i][2 * j + 1] = imaginary[i][j];
                }
            }
        }

        return complexAmp;
    }

    /**
     * Extracts the real part of a complex array.
     *
     * @param a complex array
     * @return real array
     */
    public static float[][] real(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        float[][] real = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                real[i][j] = a[i][2 * j];
            }
        }

        return real;
    }

    /**
     * Extracts the real part of a complex array.
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
     * Extracts the imaginary part of a complex array.
     *
     * @param a complex array
     * @return imaginary array
     */
    public static float[][] imaginary(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        float[][] imaginary = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                imaginary[i][j] = a[i][2 * j + 1];
            }
        }

        return imaginary;
    }

    /**
     * Extracts the imaginary part of a complex array.
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
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param a complex array
     */
    public static void complexShift(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length / 2;

        int M2 = M / 2;
        int N2 = N / 2;

        float tmp;

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
     * Performs the circular shifting of a complex array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
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
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param a real array
     */
    public static void realShift(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        int M2 = M / 2;
        int N2 = N / 2;

        float tempShift;

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
     * Performs the circular shifting of a real array, leaving the result in
     * {@code a}.
     * <p>
     * {@code
     * a b -&gt; d c
     * c d       b a
     * }
     *
     * @param a real array
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
     * Computes the pointwise multiplication of an array by {@code num}, leaving
     * the result in {@code a}.
     *
     * @param a array
     * @param num number
     *
     * @since JDiffraction 1.2
     */
    public static void multiply(float[][] a, float num) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] * num;
            }
        }
    }

    /**
     * Computes the pointwise multiplication of an array by {@code num}, leaving
     * the result in {@code a}.
     *
     * @param a array
     * @param num number
     *
     * @since JDiffraction 1.2
     */
    public static void multiply(double[][] a, double num) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] * num;
            }
        }
    }

    /**
     * Computes the pointwise division of an array by {@code num}, leaving the
     * result in {@code a}.
     *
     * @param a array
     * @param num number
     *
     * @since JDiffraction 1.2
     */
    public static void divide(float[][] a, float num) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] / num;
            }
        }
    }

    /**
     * Computes the pointwise division of an array by {@code num}, leaving the
     * result in {@code a}.
     *
     * @param a array
     * @param num number
     *
     * @since JDiffraction 1.2
     */
    public static void divide(double[][] a, double num) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] / num;
            }
        }
    }

    /**
     * Computes log10 of a real array.
     *
     * @param a array
     * @return array containing log10(a)
     */
    public static float[][] log10(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        float[][] b = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                b[i][j] = (float) Math.log10(a[i][j]);
            }
        }
        return b;
    }

    /**
     * Computes log10 of a real array.
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
     * Gets the max value of a real array.
     *
     * @param a array
     * @return max
     */
    public static float max(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        float max = a[0][0];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                max = Math.max(max, a[i][j]);
            }
        }
        return max;
    }

    /**
     * Gets the max value of a real array.
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
     * Gets the min value of a real array.
     *
     * @param a array
     * @return min
     */
    public static float min(float[][] a) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;
        float min = a[0][0];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                min = Math.min(min, a[i][j]);
            }
        }
        return min;
    }

    /**
     * Gets the min value of a real array.
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
     * Scales a real array to {@code [0, maxScale]}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public static float[][] scale(float[][] a, float max, float min, float maxScale) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        float[][] scaled = new float[M][N];

        float delta = max - min;

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
     * Scales a real array to {@code [0, maxScale]}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     * @return scaled array
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
     * Scales a real array to {@code [0, maxScale]}. Array's max and min values
     * are found using {@link #max(float[][])} and {@link #min(float[][])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public static float[][] scale(float[][] a, float maxScale) {
        float max = max(a);
        float min = min(a);

        return scale(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0, maxScale]}. Array's max and min values
     * are found using {@link #max(double[][])} and {@link #min(double[][])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     * @return scaled array
     */
    public static double[][] scale(double[][] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        return scale(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0, maxScale]} leaving the result in
     * {@code a}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
     */
    public static void scale2(float[][] a, float max, float min, float maxScale) {
        checkDimension(a);
        int M = a.length;
        int N = a[0].length;

        float delta = max - min;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = a[i][j] - min;
                a[i][j] = a[i][j] / delta;
                a[i][j] = a[i][j] * maxScale;
            }
        }
    }

    /**
     * Scales a real array to {@code [0, maxScale]} leaving the result in
     * {@code a}.
     *
     * @param a array
     * @param max array's max value
     * @param min array's min value
     * @param maxScale max value of the output array
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
     * Scales a real array to {@code [0, maxScale]} leaving the result in
     * {@code a}. Array's max and min values are found using
     * {@link #max(float[][])} and {@link #min(float[][])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     */
    public static void scale2(float[][] a, float maxScale) {
        float max = max(a);
        float min = min(a);

        scale2(a, max, min, maxScale);
    }

    /**
     * Scales a real array to {@code [0, maxScale]} leaving the result in
     * {@code a}. Array's max and min values are found using
     * {@link #max(double[][])} and {@link #min(double[][])}.
     *
     * @param a array
     * @param maxScale max value of the output array
     */
    public static void scale2(double[][] a, double maxScale) {
        double max = max(a);
        double min = min(a);

        scale2(a, max, min, maxScale);
    }

    /**
     * Converts a 1D array into a 2D array leaving the result in {@code b}. It
     * is assumed that the information on the 1D array is distributed as the
     * rows of the 2D array in sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 1D array
     * @param b 2D array
     */
    public static void vectorToMatrixArray(int M, int N, float[] a, float[][] b) {
        checkDimension(b);

        if (a.length == 0 || a.length != (M * N) || b.length != M || b[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a, N * i, b[i], 0, N);
        }
    }

    /**
     * Converts a 1D array into a 2D array leaving the result in {@code b}. It
     * is assumed that the information on the 1D array is distributed as the
     * rows of the 2D array in sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 1D array
     * @param b 2D array
     */
    public static void vectorToMatrixArray(int M, int N, double[] a, double[][] b) {
        checkDimension(b);

        if (a.length == 0 || a.length != (M * N) || b.length != M || b[0].length != N) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a, N * i, b[i], 0, N);
        }
    }

    /**
     * Converts a 2D array into a 1D array leaving the result in {@code b}. The
     * information on the 1D array is distributed as the rows of the 2D array in
     * sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 2D array
     * @param b 1D array
     */
    public static void matrixToVectorArray(int M, int N, float[][] a, float[] b) {
        checkDimension(a);

        if (a.length != M || a[0].length != N || b.length == 0 || b.length != (M * N)) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a[i], 0, b, N * i, N);
        }
    }

    /**
     * Converts a 2D array into a 1D array leaving the result in {@code b}. The
     * information on the 1D array is distributed as the rows of the 2D array in
     * sequence.
     *
     * @param M number of data points in x direction
     * @param N number of data points in y direction
     * @param a 2D array
     * @param b 1D array
     */
    public static void matrixToVectorArray(int M, int N, double[][] a, double[] b) {
        checkDimension(a);

        if (a.length != M || a[0].length != N || b.length == 0 || b.length != (M * N)) {
            throw new IllegalArgumentException("The number of data points in both arrays must be equal and different from zero.");
        }

        for (int i = 0; i < M; i++) {
            System.arraycopy(a[i], 0, b, N * i, N);
        }
    }
}
